#!/usr/bin/env python

from tkinter import Scale
import rospy
import math
import os
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseWithCovarianceStamped as Pose
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
# KF coded by yourself
from kalman_filter import KalmanFilter
import scipy.linalg as linalg


def rotate_mat(axis, radian):
    rot_matrix = linalg.expm(
        np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    return rot_matrix


class Fusion:
    def __init__(self):
        # given z (observation)
        rospy.Subscriber('/gps', Pose, self.gpsCallback)
        rospy.Subscriber('/radar_odometry', Odometry,
                         self.odometryCallback)  # given x y z
        rospy.Subscriber('/gt_odom', Odometry, self.gtCallback)  # ground truth
        rospy.on_shutdown(self.shutdown)
        self.posePub = rospy.Publisher('/pred', Odometry, queue_size=10)
        self.KF = None
        self.step = 0   # Record update times

        self.last_odometry_position = np.zeros(2)
        self.last_odometry_angle = 0

        self.gt_list = []
        self.est_list = []
        self.angel = 0
        self.origin_point = [0, 0]
        self.before_updated = [0, 0]
        self.after_updated = [0, 0]

    def shutdown(self):
        print("shuting down fusion.py")

    def gpsCallback(self, data):
        self.step += 1
        # Get GPS data only for 2D (x, y)
        measurement = np.array(
            [data.pose.pose.position.x, data.pose.pose.position.y])
        gps_covariance = np.array(data.pose.covariance).reshape(6, 6)[:2, :2]

        # KF update
        if self.step == 1:
            self.init_KF(measurement[0], measurement[1], 0)
        else:
            # original gps_covariance's diagnal are 3 ,it's too large, so i change it by 10 ^ (-2)
            self.KF.R = gps_covariance * pow(10, -2)
            # 將原本的位置(x,y)存下來
            self.before_updated = [self.KF.x[0], self.KF.x[1]]
            self.KF.update(z=[measurement[0], measurement[1]])
            # 將更新的位置(x,y)存下來
            self.after_updated = [self.KF.x[0], self.KF.x[1]]
        # 當經過15步後，gps的偏差會太大(因為不斷累加),故要進行修正
            if self.step > 15:
                temp_a = math.sqrt(
                    (self.origin_point[0]-self.after_updated[0]) ** 2 + (self.origin_point[1]-self.after_updated[1]) ** 2)
                temp_b = math.sqrt(
                    (self.before_updated[0]-self.after_updated[0]) ** 2 + (self.before_updated[1]-self.after_updated[1]) ** 2)
                temp_c = math.sqrt(
                    (self.before_updated[0]-self.origin_point[0]) ** 2 + (self.before_updated[1]-self.origin_point[1]) ** 2)
                self.angel = math.acos(
                    (temp_b * temp_b-temp_a*temp_a-temp_c*temp_c)/(-2*temp_a*temp_c))

                self.angel = -self.angel
            self.origin_point = self.after_updated

            rand_axis = [0, 0, 1]
            # get the rotation matrix
            rot_matrix = rotate_mat(rand_axis, self.angel)
            print(rot_matrix)
            self.KF.B = rot_matrix

        print(f"estimation: {self.KF.x}")

    def odometryCallback(self, data):
        self.step += 1
        # Read radar odometry data from ros msg
        position = [data.pose.pose.position.x, data.pose.pose.position.y]
        odometry_covariance = np.array(
            data.pose.covariance).reshape(6, -1)[:2, :2]

        # Get euler angle from quaternion
        roll, pitch, yaw = euler_from_quaternion(
            [data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])

        # Calculate odometry difference
        diff = [position[i] - self.last_odometry_position[i]
                for i in range(0, len(position))]

        diff_yaw = yaw - self.last_odometry_angle

        # KF predict
        if self.step == 1:
            self.init_KF(position[0], position[1], 0)
        else:
            self.KF.Q[:2, :2] = [[odometry_covariance[0][0] * pow(10, 5), 0],
                                 [0, odometry_covariance[1][1] * pow(10, 5)]]
            # self.KF.Q[:2, :2] = np.eye(2)
            self.KF.predict(u=[diff[0], diff[1], diff_yaw])
        print(f"estimation: {self.KF.x}")
        self.last_odometry_position = position
        self.last_odometry_angle = yaw

        quaternion = quaternion_from_euler(0, 0, yaw)

        # Publish odometry with covariance
        predPose = Odometry()
        predPose.header.frame_id = 'origin'
        predPose.pose.pose.position.x = self.KF.x[0]
        predPose.pose.pose.position.y = self.KF.x[1]
        predPose.pose.pose.orientation.x = quaternion[0]
        predPose.pose.pose.orientation.y = quaternion[1]
        predPose.pose.pose.orientation.z = quaternion[2]
        predPose.pose.pose.orientation.w = quaternion[3]
        predPose.pose.covariance = [self.KF.P[0][0], self.KF.P[0][1], 0, 0, 0, 0,
                                    self.KF.P[1][0], self.KF.P[1][1], 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0]
        self.posePub.publish(predPose)

    def gtCallback(self, data):
        gt_position = np.array(
            [data.pose.pose.position.x, data.pose.pose.position.y])
        self.gt_list.append(gt_position)
        if self.KF is not None:
            kf_position = self.KF.x[:2]
            self.est_list.append(kf_position)

    def plot_path(self):
        plt.figure(figsize=(10, 8))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        gt_x, gt_y = zip(*self.gt_list)
        est_x, est_y = zip(*self.est_list)
        plt.plot(gt_x, gt_y, alpha=0.25, linewidth=8, label='Groundtruth path')
        plt.plot(est_x, est_y, alpha=0.5, linewidth=3, label='Estimation path')
        plt.title("KF fusion odometry result comparison")
        plt.legend()
        if not os.path.exists("./results"):
            os.mkdir("results")
        plt.savefig("./results/result.png")
        plt.show()

    def init_KF(self, x, y, yaw):
        # Initialize the Kalman filter when the first data comes in
        self.KF = KalmanFilter(x=x, y=y, yaw=yaw)
        self.KF.A = np.identity(3)
        self.KF.B = np.identity(3)


if __name__ == '__main__':
    rospy.init_node('kf', anonymous=True)
    fusion = Fusion()
    rospy.spin()
    fusion.plot_path()
