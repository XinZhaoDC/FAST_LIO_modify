#include <mutex>
#include <nav_msgs/Odometry.h>
#include <opencv2/opencv.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <queue>
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/PointCloud2.h>
#include <thread>

#include "common_lib.h"

using namespace std;

typedef pcl::PointXYZI PointXYZI;
typedef pcl::PointCloud<PointXYZI> IPointCloud;
typedef pcl::PointCloud<PointXYZI>::Ptr IPointCloudPtr;

typedef pcl::PointXYZRGB PointXYZRGB;
typedef pcl::PointCloud<PointXYZRGB> RGBPointCloud;
typedef pcl::PointCloud<PointXYZRGB>::Ptr RGBPointCloudPtr;

std::queue<nav_msgs::OdometryConstPtr> odometryBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> undistortLaserCloudBuf;
std::queue<sensor_msgs::CompressedImageConstPtr> imageBuf;
std::mutex bufMutex;

std::vector<RGBPointCloud> vector_rgb_scan;
ros::Publisher pubColorScan;

Eigen::Matrix4d T_ItoC;

bool color_pcd_save=false;
std::string color_pcd_path;
std::string undistort_lid_topic;
std::string odo_topic;
std::string iamge_topic;
vector<double> t_LtoC(3,0.0);
vector<double> R1_LtoC(9,0.0);
vector<double> angle2_LtoC(3,0.0);

vector<double> t_LtoI(3,0.0);
vector<double> R_LtoI(9,0.0);

V3D t_Lidar_to_Camera(Zero3d);
M3D R1_Lidar_to_Camera(Eye3d);
//double theta2_Lidar_to_Camera=0.0;
V3D angle2_Lidar_to_Camera(Zero3d);
M3D R2_Lidar_to_Camera(Eye3d);
M3D R_Lidar_to_Camera(Eye3d);

V3D t_Lidar_to_IMU(Zero3d);
M3D R_Lidar_to_IMU(Eye3d);

M3D T_IMU_to_Camera(Eye3d);
 

Eigen::Matrix3d euler2RotationMatrix(const double roll, const double pitch, const double yaw)  
{  
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitZ());  
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitY());  
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitX());  
  
    Eigen::Quaterniond q = rollAngle * yawAngle * pitchAngle;  

    // Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;
    Eigen::Matrix3d R = q.matrix();  
    cout << "Euler2RotationMatrix result is:" <<endl;  
    cout << "R2 ="  << endl << R << endl<<endl;  
    return R;  
} 

/// @brief Transform point with given pose
///
/// \param[in] pt_in in pcl::Point* format
/// \param[in] pose in Eigen::Matrix4d format
/// \param[out] pt_out in pcl::Point* format
template<typename T>
inline void transformPoint(T &pt_in, Eigen::Matrix4d pose, T &pt_out) {
    pt_out = pt_in;///first copy
    Eigen::Vector4d pt(pt_in.x, pt_in.y, pt_in.z, 1);
    auto pt_world = pose * pt;
    pt_out.x = pt_world[0];
    pt_out.y = pt_world[1];
    pt_out.z = pt_world[2];
}

/// @brief Transform pointcloud with given pose
///
/// @param[in] pointcloud_in in pcl::PointCloud*::Ptr format
/// @param[in] pose in Eigen::Matrix4d format
/// @param[out] pointcloud_out in pcl::PointCloud*::Ptr format
template<typename T>
inline void transformPointcloud(T pointcloud_in, Eigen::Matrix4d pose, T pointcloud_out) {
    pointcloud_out->resize(pointcloud_in->points.size());
    for (int j = 0; j < pointcloud_in->points.size(); ++j) {
        transformPoint(pointcloud_in->points[j], pose, pointcloud_out->points[j]);
    }
}

//colorCloud(T_ItoC, 1,
//                               cv::imdecode(imageBuf.front()->data, cv::IMREAD_COLOR),
//                               laserCloudFullRes, rgb_cloud);

///Color the point cloud by rgb image using given extrinsic
void colorCloud(const Eigen::Matrix4d &extrinsic, const int density, const cv::Mat &rgb_img,
                const pcl::PointCloud<pcl::PointXYZI>::Ptr &lidar_cloud,
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr &color_cloud) {
    std::vector<cv::Point3f> pts_3d;
    for (size_t i = 0; i < lidar_cloud->size(); i += density) {
        pcl::PointXYZI point = lidar_cloud->points[i];
        float depth = sqrt(pow(point.x, 2) + pow(point.y, 2) + pow(point.z, 2));
        if (depth > 2.5 && depth < 50) {
            pts_3d.emplace_back(cv::Point3f(point.x, point.y, point.z));
        }
    }
    Eigen::AngleAxisd rotation_vector3(extrinsic.block<3, 3>(0, 0));

    //Todo write camera intrinsic and distortion into launch file

    //    cv::Mat camera_matrix =
    //            (cv::Mat_<double>(3, 3) << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0);
    //    cv::Mat distortion_coeff =
    //            (cv::Mat_<double>(1, 5) << k1_, k2_, p1_, p2_, k3_);
    // cv::Mat camera_matrix =
    //         (cv::Mat_<double>(3, 3) << 1205.698, 0.0, 1019.587, 0.0, 1205.539, 807.568, 0.0, 0.0, 1.0);
    // cv::Mat distortion_coeff =
    //         (cv::Mat_<double>(1, 5) << -0.091833, 0.082347, 0.00038, 0.00035, 0.000000);
    // cv::Mat camera_matrix =
    //         (cv::Mat_<double>(3, 3) << 335.98935910072163, 0.0, 321.54095972138805, 0.0, 337.72547533979923, 242.29014923895363, 0.0, 0.0, 1.0);
    // cv::Mat distortion_coeff =
    //         (cv::Mat_<double>(1, 5) << 0.0295468362381083, -0.0368281839741105, 0.0003400102248715159, 0.0027803141687075144, 0.0);
    //cv::Mat camera_matrix =
    //        (cv::Mat_<double>(3, 3) << 337.0953202882733, 0.0, 320.82946369947115, 0.0, 338.6261023193472, 241.39493263182715, 0.0, 0.0, 1.0);
    //cv::Mat distortion_coeff =
    //        (cv::Mat_<double>(1, 5) << 0.027465858974681987, -0.03202877927009086, -0.00011069561591140865, 0.0009081003185256085, 0.0);
    cv::Mat camera_matrix =
            (cv::Mat_<double>(3, 3) << 600.4923, 0.5563, 622.7455, 0.0, 601.001, 311.2729, 0.0, 0.0, 1.0);
    cv::Mat distortion_coeff =
            (cv::Mat_<double>(1, 5) << 0.0033, -0.0211, 0.0066, -0.0023, -0.0018);
    cv::Mat r_vec =
            (cv::Mat_<double>(3, 1)
                     << rotation_vector3.angle() * rotation_vector3.axis().transpose()[0],
             rotation_vector3.angle() * rotation_vector3.axis().transpose()[1],
             rotation_vector3.angle() * rotation_vector3.axis().transpose()[2]);
    cv::Mat t_vec = (cv::Mat_<double>(3, 1) << extrinsic(0, 3), extrinsic(1, 3), extrinsic(2, 3));
    std::vector<cv::Point2f> pts_2d;
    cv::projectPoints(pts_3d, r_vec, t_vec, camera_matrix, distortion_coeff, pts_2d);
    int image_rows = rgb_img.rows;
    int image_cols = rgb_img.cols;
    color_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (size_t i = 0; i < pts_2d.size(); i++) {
        if (pts_2d[i].x > 1 && pts_2d[i].x < image_cols - 1 &&
            pts_2d[i].y > 1 && pts_2d[i].y < image_rows - 1) {
            cv::Scalar color = rgb_img.at<cv::Vec3b>(pts_2d[i]);
            if (color[0] == 0 && color[1] == 0 && color[2] == 0) {
                continue;
            }
            pcl::PointXYZRGB p;
            p.x = pts_3d[i].x;
            p.y = pts_3d[i].y;
            p.z = pts_3d[i].z;
            p.b = color[0];
            p.g = color[1];
            p.r = color[2];
            color_cloud->points.push_back(p);
        }
    }
    color_cloud->width = color_cloud->points.size();
    color_cloud->height = 1;
}

void process() {
    ros::Rate rate(10);
    while (ros::ok()) {
        bufMutex.lock();///lock before access Buf
        if (!odometryBuf.empty() && !undistortLaserCloudBuf.empty() && !imageBuf.empty()) {

            if (fabs(odometryBuf.front()->header.stamp.toSec() -
                     undistortLaserCloudBuf.front()->header.stamp.toSec()) < std::numeric_limits<double>::epsilon()) {
                //                printf("odometry time:%f\n",odometryBuf.front()->header.stamp.toSec());
                //                printf("laser time:%f\n",undistortLaserCloudBuf.front()->header.stamp.toSec());

                IPointCloudPtr laserCloudFullRes(new IPointCloud());
                pcl::fromROSMsg(*undistortLaserCloudBuf.front(), *laserCloudFullRes);

                //                ROS_INFO("point size: %d\n", laserCloudFullRes->points.size());

                if (fabs(odometryBuf.front()->header.stamp.toSec() -
                         imageBuf.front()->header.stamp.toSec()) < 0.05) {
                    ///color points
                    //                    printf("image time:%f\n",imageBuf.front()->header.stamp.toSec());
                    RGBPointCloudPtr rgb_cloud(new RGBPointCloud());

                    colorCloud(T_ItoC, 1,
                               cv::imdecode(imageBuf.front()->data, cv::IMREAD_COLOR),
                               laserCloudFullRes, rgb_cloud);

                    //                    std::cout << "rgb cloud size: " << rgb_cloud->points.size() << std::endl;

                    RGBPointCloudPtr rgb_cloud_transformed(new RGBPointCloud());

                    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
                    pose.block<3, 3>(0, 0) = Eigen::Quaterniond(odometryBuf.front()->pose.pose.orientation.w,
                                                                odometryBuf.front()->pose.pose.orientation.x,
                                                                odometryBuf.front()->pose.pose.orientation.y,
                                                                odometryBuf.front()->pose.pose.orientation.z)
                                                     .normalized()
                                                     .toRotationMatrix();
                    pose.block<3, 1>(0, 3) = Eigen::Vector3d(odometryBuf.front()->pose.pose.position.x,
                                                             odometryBuf.front()->pose.pose.position.y,
                                                             odometryBuf.front()->pose.pose.position.z);

                    transformPointcloud(rgb_cloud, pose, rgb_cloud_transformed);

                    vector_rgb_scan.push_back(*rgb_cloud_transformed);

                    sensor_msgs::PointCloud2 colorlaserCloudMsg;
                    pcl::toROSMsg(*rgb_cloud_transformed, colorlaserCloudMsg);
                    //colorlaserCloudMsg.header.frame_id = "map";
                    colorlaserCloudMsg.header.frame_id="camera_init";
                    colorlaserCloudMsg.header.stamp = odometryBuf.front()->header.stamp;
                    std::cout << "------------------------------------------------------------------" <<std::endl;
                    pubColorScan.publish(colorlaserCloudMsg);
                    std::cout << "pub one rgb scan, rgb scan size is " << vector_rgb_scan.size() << std::endl;

                    undistortLaserCloudBuf.pop();
                    odometryBuf.pop();
                    imageBuf.pop();
                } else {
                    if (odometryBuf.front()->header.stamp.toSec() <
                        imageBuf.front()->header.stamp.toSec()) {
                        //                        std::cout << "odometry time small image time" << std::endl;
                        odometryBuf.pop();
                        //undistortLaserCloudBuf.pop();
                        bufMutex.unlock();
                        continue;
                    }
                    if (odometryBuf.front()->header.stamp.toSec() >
                        imageBuf.front()->header.stamp.toSec()) {
                        //                        std::cout << "odometry time large image time" << std::endl;
                        imageBuf.pop();
                        bufMutex.unlock();
                        continue;
                    }
                }
            } else {

                if (odometryBuf.front()->header.stamp.toSec() <
                    undistortLaserCloudBuf.front()->header.stamp.toSec()) {
                    //                    std::cout << "odometry time small lidar time" << std::endl;
                    //                    printf("odometry time:%f\n",odometryBuf.front()->header.stamp.toSec());
                    //                    printf("laser time:%f\n",undistortLaserCloudBuf.front()->header.stamp.toSec());
                    odometryBuf.pop();
                    bufMutex.unlock();
                    continue;
                }
                if (odometryBuf.front()->header.stamp.toSec() >
                    undistortLaserCloudBuf.front()->header.stamp.toSec()) {
                    //                    std::cout << "odometry time large lidar time" << std::endl;
                    //                    printf("odometry time:%f\n",odometryBuf.front()->header.stamp.toSec());
                    //                    printf("laser time:%f\n",undistortLaserCloudBuf.front()->header.stamp.toSec());
                    undistortLaserCloudBuf.pop();
                    bufMutex.unlock();
                    continue;
                }
            }
        }
        bufMutex.unlock();

        rate.sleep();
    }
}

void compressedImageHandler(const sensor_msgs::CompressedImageConstPtr &msgCompressedImage) {
    bufMutex.lock();
    //    ROS_INFO("receive image");
    imageBuf.push(msgCompressedImage);
    bufMutex.unlock();
}

void laserOdometryHandler(const nav_msgs::OdometryConstPtr &msgLaserOdom) {
    bufMutex.lock();
    //    ROS_INFO("receive odometry");
    odometryBuf.push(msgLaserOdom);
    bufMutex.unlock();
}

void laserCloudUndistortHandler(const sensor_msgs::PointCloud2ConstPtr &msgLaserCloudUndistort) {
    bufMutex.lock();
    //    ROS_INFO("receive laser cloud");
    undistortLaserCloudBuf.push(msgLaserCloudUndistort);
    bufMutex.unlock();
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "colorPointclouds");
    ros::NodeHandle nh;

     nh.param<bool>("pcd_save/color_pcd_save",color_pcd_save, false);
     nh.param<std::string>("pcd_save/color_pcd_path",color_pcd_path,"./color_pcd");
     nh.param<std::string>("publish/undistort_lid_topic",undistort_lid_topic,"/cloud_registered");
     nh.param<std::string>("publish/odometry",odo_topic,"/Odometry");
     nh.param<std::string>("common/image_topic",iamge_topic,"/camera/image/compressed");
     nh.param<vector<double>>("extrinsic_LtoC/t_LtoC", t_LtoC, vector<double>());
     nh.param<vector<double>>("extrinsic_LtoC/R1_LtoC", R1_LtoC, vector<double>());
     nh.param<vector<double>>("extrinsic_LtoC/angle2_LtoC", angle2_LtoC, vector<double>());
     nh.param<vector<double>>("mapping/extrinsic_T", t_LtoI, vector<double>());
     nh.param<vector<double>>("mapping/extrinsic_R", R_LtoI, vector<double>());

    ros::Subscriber subLaserCloudUndistort = nh.subscribe<sensor_msgs::PointCloud2>(
            undistort_lid_topic,
            //"/intercept_cloud",
            // "/cloud_registered",
            100,
            laserCloudUndistortHandler);

    ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>(
            odo_topic,
            //"/Odometry",
            100,
            laserOdometryHandler);

    ros::Subscriber subCompressedImage = nh.subscribe<sensor_msgs::CompressedImage>(
            iamge_topic,
            //"/jackal/camera_f/compressed",
            100,
            compressedImageHandler);

    pubColorScan = nh.advertise<sensor_msgs::PointCloud2>("/color_scan", 10);

    ///undistorted pointclous is in body(imu) frame, so we use T_LtoC and T_LtoI get T_ItoC
    //Todo write into launch file
    // Eigen::Matrix4d T_LtoC, T_LtoI;
    // T_LtoI << 0.00847397, -0.999944, 0.00626681, -0.0824134,
    //         0.999964, 0.00847539, 0.000198841, 0.095912,
    //         -0.000251944, 0.0062649, 0.99998, 0.0102022,
    //         0, 0, 0, 1;

    // T_LtoC << 0.00920662, -0.999907, 0.0101051, -0.057,
    //         -0.0214086, -0.0103003, -0.999718, 0.028,
    //         0.999728, 0.00898768, -0.0215014, 0,
    //         0, 0, 0, 1;

    // T_ItoC = T_LtoC * T_LtoI.inverse();

    // T_ItoC << 1, 0, 0, -0.2,
    //         0, 1, 0, 0,
    //         0, 0, 1, 0.1,
    //         0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000;

    /*V3D t_Lidar_to_Camera(Zero3d);
M3D R1_Lidar_to_Camera(Eye3d);

M3D R_Lidar_to_Camera(Eye3d);*/
    t_Lidar_to_Camera<<VEC_FROM_ARRAY(t_LtoC);
    R1_Lidar_to_Camera<<MAT_FROM_ARRAY(R1_LtoC);
    angle2_Lidar_to_Camera<<VEC_FROM_ARRAY(angle2_LtoC);

    t_Lidar_to_IMU<<VEC_FROM_ARRAY(t_LtoI);
    R_Lidar_to_IMU<<MAT_FROM_ARRAY(R_LtoI);

    //std::cout<<"R1_Lidar_to_Camera:"<<std::endl;
    //std::cout<<R1_Lidar_to_Camera<<std::endl;

    //std::cout<<"R_Lidar_to_IMU:"<<std::endl;
    //std::cout<<R_Lidar_to_IMU<<std::endl;



    //Z-Roll，X-Pitch，Y-Yaw
    R2_Lidar_to_Camera=euler2RotationMatrix(angle2_Lidar_to_Camera(0)*3.1415926/180,angle2_Lidar_to_Camera(1)*3.1415926/180,angle2_Lidar_to_Camera(2)*3.1415926/180);
    std::cout<<"R2_Lidar_to_Camera:"<<std::endl;
    std::cout<<R2_Lidar_to_Camera<<std::endl;
    //R2_Lidar_to_Camera(0,0)=1.0;
    //R2_Lidar_to_Camera(0,1)=0.0;
    //R2_Lidar_to_Camera(0,2)=0.0;
    //R2_Lidar_to_Camera(1,0)=0.0;
    //R2_Lidar_to_Camera(1,1)=cos(3.1415926*theta2_LtoC/180);
    //R2_Lidar_to_Camera(1,2)=-sin(3.1415926*theta2_LtoC/180);
    //R2_Lidar_to_Camera(2,0)=0.0;
    //R2_Lidar_to_Camera(2,1)=sin(3.1415926*theta2_LtoC/180);
    //R2_Lidar_to_Camera(2,2)=cos(3.1415926*theta2_LtoC/180);




    //std::cout<<"R2_Lidar_to_Camera:"<<std::endl;
    //std::cout<<R2_Lidar_to_Camera<<std::endl;
    //for(int i=0; i<3; i++){
    //    for(int j=0; j<3; j++){
    //        std::cout<<R2_Lidar_to_Camera(i,j)<<"       ";
    //    }
    //    std::cout<<"\n";
    //}

    

    R_Lidar_to_Camera=R2_Lidar_to_Camera*R1_Lidar_to_Camera;
    std::cout<<"R_Lidar_to_Camera:"<<std::endl;
    std::cout<<R_Lidar_to_Camera<<std::endl;

    Eigen::Matrix4d T_Lidar_to_Camera=Eigen::Matrix4d::Identity();
    T_Lidar_to_Camera.block<3,3>(0,0)=R_Lidar_to_Camera;
    T_Lidar_to_Camera.block<3,1>(0,3)=t_Lidar_to_Camera;
    std::cout<<"T_Lidar_to_Camera:"<<std::endl;
    std::cout<<T_Lidar_to_Camera<<std::endl;

    Eigen::Matrix4d T_Lidar_to_IMU=Eigen::Matrix4d::Identity();
    T_Lidar_to_IMU.block<3,3>(0,0)=R_Lidar_to_IMU;
    T_Lidar_to_IMU.block<3,1>(0,3)=t_Lidar_to_IMU;
    std::cout<<"T_Lidar_to_IMU:"<<std::endl;
    std::cout<<T_Lidar_to_IMU<<std::endl;

    T_ItoC=T_Lidar_to_Camera*T_Lidar_to_IMU.inverse();
    std::cout<<"T_IMU_to_Camera:"<<std::endl;
    std::cout << T_ItoC << std::endl;

    /*Eigen::Matrix3d init_camera_r;
    double init_xyz[3] = {0.1,0.28,-0.08};
    init_camera_r = euler2RotationMatrix((1.57 - 3.1415926*1.2/180),-3.1415926*0/180,-(1.57 + 3.1415926*7/180));

    for (int cti = 0 ; cti < 3 ;cti ++)
    {
        for(int ctj = 0; ctj < 3 ; ctj ++)
        {
            T_ItoC(cti , ctj) = init_camera_r(cti , ctj);
        }
    }
    for (int cti = 0 ; cti < 3 ;cti ++)
    {
        T_ItoC(cti , 3) = init_xyz[cti];
    }
    T_ItoC(3 , 3) = 1;*/
    

//          T_ItoC <<6.34136e-07 ,         -1 ,0.000796326,-0.2,
// 0.000796326 ,0.000796327   , 0.999999,0,
//          -1, 5.55112e-17 ,0.000796327,0.1,
//             0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000;

    std::thread thread_process{process};
    ros::spin();

    ///Output rgb map

    if(color_pcd_save){
        //RGBPointCloudPtr rgb_map(new RGBPointCloud());
        //for (int i = 0; i < vector_rgb_scan.size(); ++i) {
        //    *rgb_map += vector_rgb_scan[i];
        //}
        //std::string save_color_pcd=color_pcd_path+"/color_pointcloud.pcd";
        //std::cout << "rgb_map size is " << *rgb_map.size() << std::endl;
        //pcl::io::savePCDFile(save_color_pcd, *rgb_map);

        
        for (int i = 0; i < vector_rgb_scan.size(); ++i) {
            RGBPointCloudPtr rgb_map(new RGBPointCloud());
            *rgb_map = vector_rgb_scan[i];
            std::string save_color_pcd=color_pcd_path+"/color_pointcloud_"+std::to_string(i)+".pcd";
            pcl::io::savePCDFile(save_color_pcd, *rgb_map);
        }
        
    }
    
}
