#!/usr/bin/env python

from tkinter import Y, Scale
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
from Kalman_filter import KalmanFilter
import scipy.linalg as linalg


################################################
origin_point = []


def cal_ang(point_1, point_2, point_3):
    # """
    # :param point_1: 点1坐标
    # :param point_2: 点2坐标
    # :param point_3: 点3坐标
    # """
    a = math.sqrt((point_2[0]-point_3[0])*(point_2[0]-point_3[0]) +
                  (point_2[1]-point_3[1])*(point_2[1] - point_3[1]))
    b = math.sqrt((point_1[0]-point_3[0])*(point_1[0]-point_3[0]) +
                  (point_1[1]-point_3[1])*(point_1[1] - point_3[1]))
    c = math.sqrt((point_1[0]-point_2[0])*(point_1[0]-point_2[0]) +
                  (point_1[1]-point_2[1])*(point_1[1]-point_2[1]))
    A = math.degrees(math.acos((a*a-b*b-c*c)/(-2*b*c)))
    B = math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c)))
    C = math.degrees(math.acos((c*c-a*a-b*b)/(-2*a*b)))
    return B


def rotate_mat(axis, radian):
    rot_matrix = linalg.expm(
        np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    return rot_matrix

 #####################################


class Fusion:
    def __init__(self):
        rospy.Subscriber('/gps', Pose, self.gpsCallback)
        rospy.Subscriber('/radar_odometry', Odometry, self.odometryCallback)
        rospy.Subscriber('/gt_odom', Odometry, self.gtCallback)
        rospy.on_shutdown(self.shutdown)
        self.posePub = rospy.Publisher('/pred', Odometry, queue_size=10)
        self.KF = None
        self.step = 0   # Record update times

        self.last_odometry_position = np.zeros(2)
        self.last_odometry_angle = 0

        self.gt_list = []
        self.est_list = []

    def shutdown(self):
        print("shuting down fusion.py")

    def gpsCallback(self, data):
        self.step += 1
        global origin_point
        before_updated = []
        after_updated = []

        # Get GPS data only for 2D (x, y)
        measurement = np.array(
            [data.pose.pose.position.x, data.pose.pose.position.y])
        gps_covariance = np.array(data.pose.covariance).reshape(6, 6)[:2, :2]
        print("gps", gps_covariance)
        # gps的更新時間為1hz   how to know it
        print(self.step)
        # KF update
        if self.step == 1:
            self.init_KF(measurement[0], measurement[1], 0)
        else:
            self.KF.Q = [[0.05, 0], [0, 0.05]]

            before_updated.append([self.KF.x[0], self.KF.x[1]])
            print("before updated:", before_updated)
            print("#################################################################################################")

            # update is to update the x (position) value
            self.KF.update(z=[measurement[0], measurement[1]])

            # record the updated gps value
            after_updated.append([self.KF.x[0], self.KF.x[1]])
            print("after updated:", after_updated)
            print("#################################################################################################")

            if self.step > 15:
                angel = cal_ang((before_updated[0][0], before_updated[0][1]), (
                    origin_point[0][0], origin_point[0][1]), (after_updated[0][0], after_updated[0][1]))
                angel = -angel
                print(angel)
                print(
                    "#################################################################################################")

            origin_point = after_updated
            rand_axis = [0, 0, 1]
            # rotate angel
            angel = math.radians(angel)
            # get the rotation matrix
            rot_matrix = rotate_mat(rand_axis, angel)
            print(rot_matrix)
            self.KF.B = rot_matrix

        print(f"estimation: {self.KF.x}")  # 一直更新

    def odometryCallback(self, data):
        self.step += 1
        # Read radar odometry data from ros msg
        position = [data.pose.pose.position.x, data.pose.pose.position.y]
        odometry_covariance = np.array(
            data.pose.covariance).reshape(6, -1)[:2, :2]

        # Get euler angle from quaternion
        roll, pitch, yaw = euler_from_quaternion([data.pose.pose.orientation.x,
                                                  data.pose.pose.orientation.y,
                                                  data.pose.pose.orientation.z,
                                                  data.pose.pose.orientation.w])
        print("odo", odometry_covariance)

        # Calculate odometry difference
        # diff = [data.pose.pose.orientation.x - self.last_odometry_position[0],
        #         data.pose.pose.orientation.y - self.last_odometry_position[1]]
        diff = [position[0] - self.last_odometry_position[0],
                position[1] - self.last_odometry_position[1]]
        # print("diff",diff)
        diff_yaw = yaw - self.last_odometry_angle

        # KF predict
        if self.step == 1:
            self.init_KF(position[0], position[1], 0)
        else:
            self.KF.R[:2, :2] = [[0.5, 0], [0, 0.5]]
            # predict is to referesh the K matric
            self.KF.predict(u=[diff[0], diff[1], diff_yaw])

        print(f"estimation: {self.KF.x}")
        self.last_odometry_position = position
        self.last_odometry_angle = yaw

        quaternion = quaternion_from_euler(0, 0, yaw)

        # Publish odometry with covariance  #predict mean
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
        # print(predPose)
        self.posePub.publish(predPose)

    def gtCallback(self, data):
        gt_position = np.array(
            [data.pose.pose.position.x, data.pose.pose.position.y])
        gt_position = np.array(
            [data.pose.pose.position.x, data.pose.pose.position.y])
        self.gt_list.append(gt_position)
        if self.KF is not None:
            kf_position = self.KF.x[:2]
            # print(kf_position)
            self.est_list.append(kf_position)
        # print("estimate position",self.KF.x[:2])    #this estimate position is the same in the gps callback estimation

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
    # fusion.plot_path()
