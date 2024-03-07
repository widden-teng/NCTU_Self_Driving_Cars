#include<iostream>
#include<fstream>
#include<limits>
#include<vector>

#include<ros/ros.h>
#include<sensor_msgs/PointCloud2.h>
#include<geometry_msgs/PointStamped.h>
#include<geometry_msgs/PoseStamped.h>
#include<tf/transform_broadcaster.h>
#include<tf2_eigen/tf2_eigen.h>


#include<Eigen/Dense>

#include<pcl/registration/icp.h>
#include<pcl/filters/voxel_grid.h>
#include<pcl_conversions/pcl_conversions.h>
#include<pcl_ros/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

class Localizer{
private:

  float mapLeafSize = 1., scanLeafSize = 1.;
  std::vector<float> d_max_list, n_iter_list;

  ros::NodeHandle _nh;
  ros::Subscriber sub_map, sub_points, sub_gps;
  ros::Publisher pub_points, pub_pose, pub_test;
  tf::TransformBroadcaster br;

  pcl::PointCloud<pcl::PointXYZI>::Ptr map_points;
  pcl::PointXYZ gps_point;
  bool gps_ready = false, map_ready = false, initialied = false;
  Eigen::Matrix4f init_guess;
  int cnt = 0;
  int count = 1;
  pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
  pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;

  std::string result_save_path;
  std::ofstream outfile;
  geometry_msgs::Transform car2Lidar;
  std::string mapFrame, lidarFrame;

public:

  Localizer(ros::NodeHandle nh): map_points(new pcl::PointCloud<pcl::PointXYZI>) {
    std::vector<float> trans, rot;

    _nh = nh;
    //這邊是nodehandle內的參數設定
    _nh.param<std::vector<float>>("baselink2lidar_trans", trans, std::vector<float>());
    _nh.param<std::vector<float>>("baselink2lidar_rot", rot, std::vector<float>());
    _nh.param<std::string>("result_save_path", result_save_path, "result.csv");
    _nh.param<float>("scanLeafSize", scanLeafSize, 1.0);
    _nh.param<float>("mapLeafSize", mapLeafSize, 1.0);
    _nh.param<std::string>("mapFrame", mapFrame, "world");
    _nh.param<std::string>("lidarFrame", lidarFrame, "nuscenes_lidar");

    // c_str() 將string轉為c字符串的形式
    ROS_INFO("saving results to %s", result_save_path.c_str());
    //將資料寫入
    outfile.open(result_save_path);
    outfile << "id,x,y,z,yaw,pitch,roll" << std::endl;

    //trans 跟下面一樣
    if(trans.size() != 3 | rot.size() != 4){
      ROS_ERROR("transform not set properly");
    }
    // at(x) 用於給予vector於x位置中的值
    car2Lidar.translation.x = trans.at(0);
    car2Lidar.translation.y = trans.at(1);
    car2Lidar.translation.z = trans.at(2);
    car2Lidar.rotation.x = rot.at(0);
    car2Lidar.rotation.y = rot.at(1);
    car2Lidar.rotation.z = rot.at(2);
    car2Lidar.rotation.w = rot.at(3);

    //this 才知道是在這個class內的function
    sub_map = _nh.subscribe("/map", 1, &Localizer::map_callback, this);
    sub_points = _nh.subscribe("/lidar_points", 400, &Localizer::pc_callback, this);
    sub_gps = _nh.subscribe("/gps", 1, &Localizer::gps_callback, this);
    pub_points = _nh.advertise<sensor_msgs::PointCloud2>("/transformed_points", 1);
    pub_pose = _nh.advertise<geometry_msgs::PoseStamped>("/lidar_pose", 1);
    pub_test = _nh.advertise<sensor_msgs::PointCloud2>("/test_points", 1);
    //用於初始化init_guess
    init_guess.setIdentity();
    ROS_INFO("%s initialized", ros::this_node::getName().c_str());
  }

  // Gentaly end the node
  ~Localizer(){
    if(outfile.is_open()) outfile.close();
  }
  // get the imformation of map
  void map_callback(const sensor_msgs::PointCloud2::ConstPtr& msg){
    ROS_INFO("Got map message");
    // pcl::fromROSMsg 用於 sensor_msgs::PointCloud2 和 pcl::PointCloud 之间的转换
    // pointcloud2 的資訊較pointcloud 多很多
    pcl::fromROSMsg(*msg, *map_points);
    map_ready = true;
  }
  
  void pc_callback(const sensor_msgs::PointCloud2::ConstPtr& msg){
    ROS_INFO("Got lidar message");
    pcl::PointCloud<pcl::PointXYZI>::Ptr scan_ptr(new pcl::PointCloud<pcl::PointXYZI>);
    Eigen::Matrix4f result;

    while(!(gps_ready & map_ready)){
      ROS_WARN("waiting for map and gps data ...");
      ros::Duration(0.05).sleep();
      ros::spinOnce();
    }

    pcl::fromROSMsg(*msg, *scan_ptr);
    ROS_INFO("point size: %d", scan_ptr->width);
    //result is type of Eigen::Matrix4f
    result = align_map(scan_ptr);

    // publish transformed points
    sensor_msgs::PointCloud2::Ptr out_msg(new sensor_msgs::PointCloud2);
    // Transform a sensor_msgs::PointCloud2 dataset using an Eigen 4x4 matrix.  (const Eigen::Matrix4f &transform, const sensor_msgs::PointCloud2 &in, sensor_msgs::PointCloud2 &out)
    pcl_ros::transformPointCloud(result, *msg, *out_msg);
    out_msg->header = msg->header;
    // mapFrame is "world"
    out_msg->header.frame_id = mapFrame;
    //public 到/transformed_points
    pub_points.publish(out_msg);

    // broadcast transforms
    tf::Matrix3x3 rot;
    //static_cast<T> Set the values of the matrix explicitly (row major)
    //here is for ratation matrix
    rot.setValue(
      //轉換型態，將resul(..)轉換為double
      static_cast<double>(result(0, 0)), static_cast<double>(result(0, 1)), static_cast<double>(result(0, 2)), 
      static_cast<double>(result(1, 0)), static_cast<double>(result(1, 1)), static_cast<double>(result(1, 2)),
      static_cast<double>(result(2, 0)), static_cast<double>(result(2, 1)), static_cast<double>(result(2, 2))
    );
    //for translation matrix 
    tf::Vector3 trans(result(0, 3), result(1, 3), result(2, 3));
    //TF Transform class supports rigid transforms with only translation and rotation and no scaling/shear.
    tf::Transform transform(rot, trans);
    //使mapFrame變成由lidarFrame經StampedTransform生成
    br.sendTransform(tf::StampedTransform(transform.inverse(), msg->header.stamp, lidarFrame, mapFrame));

    // publish lidar pose
    geometry_msgs::PoseStamped pose;
    pose.header = msg->header;
    pose.header.frame_id = mapFrame;
    pose.pose.position.x = trans.getX();
    pose.pose.position.y = trans.getY();
    pose.pose.position.z = trans.getZ();
    pose.pose.orientation.x = transform.getRotation().getX();
    pose.pose.orientation.y = transform.getRotation().getY();
    pose.pose.orientation.z = transform.getRotation().getZ();
    pose.pose.orientation.w = transform.getRotation().getW();
    pub_pose.publish(pose);

    //Affine3d T是一个4*4齐次矩阵变换
    Eigen::Affine3d transform_c2l, transform_m2l;
    transform_m2l.matrix() = result.cast<double>();
    //Convert a timestamped transform to the equivalent Eigen data type
    transform_c2l = (tf2::transformToEigen(car2Lidar));
    Eigen::Affine3d tf_p = transform_m2l * transform_c2l.inverse();
    geometry_msgs::TransformStamped transform_m2c = tf2::eigenToTransform(tf_p);

    tf::Quaternion q(transform_m2c.transform.rotation.x, transform_m2c.transform.rotation.y, transform_m2c.transform.rotation.z, transform_m2c.transform.rotation.w);
    //The tfScalar type abstracts floating point numbers, to easily switch between double and single floating point precision
    tfScalar yaw, pitch, roll;
    tf::Matrix3x3 mat(q);
    mat.getEulerYPR(yaw, pitch, roll);
    outfile << ++cnt << "," << tf_p.translation().x() << "," << tf_p.translation().y() << "," << tf_p.translation().z() << "," << yaw << "," << pitch << "," << roll << std::endl;
    ROS_INFO("save %d's data", count);
    count = count+1;
  }
  //use to update gps imformation
  void gps_callback(const geometry_msgs::PointStamped::ConstPtr& msg){
    ROS_INFO("Got GPS message");
    gps_point.x = msg->point.x;
    gps_point.y = msg->point.y;
    gps_point.z = msg->point.z;

    //use to set tf between nuscenes_lidar and world
    if(!initialied){
    // if(true){
      geometry_msgs::PoseStamped pose;
      pose.header = msg->header;
      pose.pose.position = msg->point;
      pub_pose.publish(pose);
      // ROS_INFO("pub pose");

      tf::Matrix3x3 rot;
      rot.setIdentity();
      tf::Vector3 trans(msg->point.x, msg->point.y, msg->point.z);
      tf::Transform transform(rot, trans);
      br.sendTransform(tf::StampedTransform(transform, msg->header.stamp, "world", "nuscenes_lidar"));
    }

    gps_ready = true;
    return;
  }
  // for scon_points is lidar_points (type is sensor_msgs::PointCloud2::ConstPtr or pcl::PointCloud)
  Eigen::Matrix4f align_map(const pcl::PointCloud<pcl::PointXYZI>::Ptr scan_points){
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_map_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr temp_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr z_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr x_near_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr x_right_near_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    
    Eigen::Matrix4f result;
    sensor_msgs::PointCloud2::Ptr test_msg(new sensor_msgs::PointCloud2);

    /* [Part 1] Perform pointcloud preprocessing here e.g. downsampling use setLeafSize(...) ... */
    //uses the base Filter class methods to pass through all data that satisfies the user given constraints
    pcl::PassThrough<pcl::PointXYZI> pass, unpass;



    //////////////////////////////////////////////////////////////////////////////////
    // z of higher than car
    pass.setInputCloud(scan_points);
    //According to x axis to do filtering
  	pass.setFilterFieldName("x");
    pass.setFilterLimits(-35.0, 35.0);
	  //pass.setFilterLimits(-20.0, 35.0); ////this has better
    pass.filter(*z_scan_ptr);
    
    //According to y axis to do filtering
    pass.setInputCloud(z_scan_ptr);
  	pass.setFilterFieldName("y");
    pass.setFilterLimits(-50.0, 50.0);
	  // pass.setFilterLimits(-25.0, 25.0);
    pass.filter(*z_scan_ptr);
    
    //According to z axis to do filtering
    pass.setInputCloud(z_scan_ptr);
    pass.setFilterFieldName("z");
	  // pass.setFilterLimits(1.0, 10.0); 
    pass.setFilterLimits(1.0, 10.0);
    pass.filter(*z_scan_ptr);
    

    *filtered_scan_ptr += *z_scan_ptr;
    // filtered_scan_ptr = z_scan_ptr;


    //////////////////////////////////////////////////////////////////////////////////
    // x near left car
    pass.setInputCloud(scan_points);
    //According to x axis to do filtering
  	pass.setFilterFieldName("x");
    pass.setFilterLimits(-6.5, 0.0); //(-5.0, 0.0)
	  //pass.setFilterLimits(10.0, 35.0); ////this has better
    pass.filter(*x_near_scan_ptr);
    
    //According to y axis to do filtering
    pass.setInputCloud(x_near_scan_ptr);
  	pass.setFilterFieldName("y");
    pass.setFilterLimits(-50.0, 50.0);
	  // pass.setFilterLimits(-25.0, 25.0);
    pass.filter(*x_near_scan_ptr);
    
    //According to z axis to do filtering
    pass.setInputCloud(x_near_scan_ptr);
    pass.setFilterFieldName("z");
	  // pass.setFilterLimits(1.0, 10.0); 
    pass.setFilterLimits(-2.0, 1.0);
    pass.filter(*x_near_scan_ptr);
    

    *filtered_scan_ptr += *x_near_scan_ptr;
    // filtered_scan_ptr = x_near_scan_ptr;


    //////////////////////////////////////////////////////////////////////////////////
    // x near right car
    pass.setInputCloud(scan_points);
    //According to x axis to do filtering
  	pass.setFilterFieldName("x");
    pass.setFilterLimits(0.0, 5.3); //(0.0, 5.5)
	  //pass.setFilterLimits(10.0, 35.0); ////this has better
    pass.filter(*x_right_near_scan_ptr);
    
    //According to y axis to do filtering
    pass.setInputCloud(x_right_near_scan_ptr);
  	pass.setFilterFieldName("y");
    pass.setFilterLimits(-50.0, 50.0);
	  // pass.setFilterLimits(-25.0, 25.0);
    pass.filter(*x_right_near_scan_ptr);
    
    //According to z axis to do filtering
    pass.setInputCloud(x_right_near_scan_ptr);
    pass.setFilterFieldName("z");
	  // pass.setFilterLimits(1.0, 10.0); 
    pass.setFilterLimits(-2.0, 1.0);
    pass.filter(*x_right_near_scan_ptr);
    

    *filtered_scan_ptr += *x_right_near_scan_ptr;
    // filtered_scan_ptr = x_near_scan_ptr;


    //////////////////////////////////////////////////////////////////////////////////
    //filter in y axis

    //According to y axis to do filtering
    pass.setInputCloud(filtered_scan_ptr);
  	pass.setFilterFieldName("y");
    pass.setFilterLimits(-25.0, 25.0);
	  // pass.setFilterLimits(-25.0, 25.0);
    pass.filter(*filtered_scan_ptr);

    //////////////////////////////////////////////////////////////////////////////////
    //filter out x axis
    // unpass.setInputCloud(filtered_scan_ptr);
    unpass.setInputCloud(filtered_scan_ptr);
    unpass.setFilterFieldName("x");
	  unpass.setFilterLimits(-5.8, 4.3); 
    unpass.setFilterLimitsNegative(true);
    unpass.filter(*filtered_scan_ptr);
    
    //use for voxel down, can also remove noise
    voxel_filter.setInputCloud (filtered_scan_ptr);
    // voxel_filter.setInputCloud (scan_points);
    //(1.0f)
    voxel_filter.setLeafSize (0.3f, 0.3f, 0.3f); //0.6f
    voxel_filter.filter (*filtered_scan_ptr);
    std::cout << "filtered_scan_ptr: " << filtered_scan_ptr->size() << std::endl;

    // if no use voxel down, map points will too much
    voxel_filter.setInputCloud (map_points);
    //(0.6f)
    voxel_filter.setLeafSize (0.3f, 0.3f, 0.3f);
    voxel_filter.filter (*filtered_map_ptr);
    std::cout << "filtered_map_ptr: " << filtered_map_ptr->size() << std::endl;

    /* Find the initial orientation for fist scan */
    if(!initialied){
      //provides a base implementation of the Iterative Closest Point algorithm
      pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> first_icp;
      //std::numeric_limits<T>::max  Returns the maximum finite value representable by the numeric type T.
      float yaw, min_yaw, min_score = std::numeric_limits<float>::max();
      Eigen::Matrix4f min_pose(Eigen::Matrix4f::Identity());
	  /* [Part 3] you can perform ICP several times to find a good initial guess */
      //find the rotation in radians, try and err
      // //////////////////////////////////////////////////////////////////////////////////////////
      // yaw = -M_PI*49/40; 
      // min_pose(0,0) = cos(yaw);
      // min_pose(0,1) = -sin(yaw);
      // min_pose(1,0) = sin(yaw);
      // min_pose(1,1) = cos(yaw);
      // min_pose(0,3) = gps_point.x;
      // min_pose(1,3) = gps_point.y;
      // min_pose(2,3) = gps_point.z;
      ///////////////////////////////////////////////////////////////////////////////////////////
      // set initial guess
      /////////////////////////////////////
      int iter = 1;
      float temp = 0.0;
      for (yaw = 0; yaw < (M_PI * 2); yaw += 0.6) {
        Eigen::Translation3f init_translation(gps_point.x, gps_point.y, gps_point.z);
        Eigen::AngleAxisf init_rotation_z(yaw, Eigen::Vector3f::UnitZ());
        init_guess = (init_translation * init_rotation_z).matrix();

        first_icp.setInputSource(filtered_scan_ptr);
        first_icp.setInputTarget(filtered_map_ptr);
        first_icp.setMaxCorrespondenceDistance(0.9);
        first_icp.setMaximumIterations(500); //5000
        first_icp.setTransformationEpsilon(1e-9);//-8
        first_icp.setEuclideanFitnessEpsilon(1e-9);//-8
        first_icp.align(*transformed_scan_ptr, init_guess);
        ROS_INFO("Updating");
        temp = std::floor((iter/((M_PI*2 -1)/0.6) * 100)*100)/100;
        std::cout<<"Updating min_yaw ....."<<temp<<"%"<<std::endl;
        double score = first_icp.getFitnessScore(0.5);
        if (score < min_score) {
            min_score = score;
            min_pose = first_icp.getFinalTransformation();
            ROS_INFO("Update best pose");
            min_yaw = yaw;
        }
        iter++;

      }
      ////////////////////////////////
      // yaw = 4.8;
      // Eigen::Translation3f init_translation(gps_point.x, gps_point.y, gps_point.z);
      // Eigen::AngleAxisf init_rotation_z(yaw, Eigen::Vector3f::UnitZ());
      // init_guess = (init_translation * init_rotation_z).matrix();

      // first_icp.setInputSource(filtered_scan_ptr);
      // first_icp.setInputTarget(filtered_map_ptr);
      // first_icp.setMaxCorrespondenceDistance(0.9);
      // first_icp.setMaximumIterations(500); //5000
      // first_icp.setTransformationEpsilon(1e-9); //-8
      // first_icp.setEuclideanFitnessEpsilon(1e-9); //-8
      // first_icp.align(*transformed_scan_ptr, init_guess);
      // min_pose = first_icp.getFinalTransformation();

     // ///////


      ROS_INFO("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
      ROS_INFO("min_yaw is : %f",min_yaw);
      init_guess = min_pose;
      initialied = true;
    }
	
	/* [Part 2] Perform ICP here or any other scan-matching algorithm */
	/* Refer to https://pointclouds.org/documentation/classpcl_1_1_iterative_closest_point.html#details */
    //要疊的部份
    icp.setInputSource(filtered_scan_ptr);
    //疊過去的部份
    icp.setInputTarget(filtered_map_ptr);
    //Set the max correspondence distance to 1cm, remove outliner 1.(0.9) 2.(20) 
    icp.setMaxCorrespondenceDistance(6); //6
    // Set the maximum number of iterations (5000)
    icp.setMaximumIterations(5000);
    //前一个变换矩阵和当前变换矩阵的差异小于阈值时，就认为已经收敛了，是一条收敛条件 (1e-09)
    icp.setTransformationEpsilon(1e-9);
    //还有一条收敛条件是均方误差和小于阈值， 停止迭代 (1e-9)
    icp.setEuclideanFitnessEpsilon(1e-9);
    //transformed_scan_ptr 為新的點雲，將source 先根據init_guess轉換後在根據icp與target 疊合所求得
    icp.align(*transformed_scan_ptr, init_guess);
    if(!icp.hasConverged ())
    {
      std::cout << "not converge" << std::endl;
    }
    std::cout << "initial_guess: " << init_guess << std::endl;
    //result is type of Eigen::Matrix4f
    result = icp.getFinalTransformation();
    std::cout << "result: " << result << std::endl;
    std::cout << "=====================================" << std::endl;
    //icp.getFitnessScore() goar of fitness
    std::cout << "The MSE is : "<< icp.getFitnessScore() << std::endl;
    // pub filtered point
    pcl::toROSMsg(*transformed_scan_ptr, *test_msg);
    test_msg->header.frame_id = "world";
    pub_test.publish(test_msg);
    std::cout << "======================================" << std::endl;
	/* Use result as next initial guess */
    init_guess = result;
    return result;
  }
};


int main(int argc, char* argv[]){
  ros::init(argc, argv, "localizer");
  ros::NodeHandle n("~");
  Localizer localizer(n);
  ros::spin();
  return 0;
}
