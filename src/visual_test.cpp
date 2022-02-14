//
// Created by joeray on 2022/2/2.
//

#include <iostream>

#include <ros/ros.h>
#include "sensor_msgs/LaserScan.h"
#include "geometry_msgs/Twist.h"
#include "geometry_msgs/PoseStamped.h"
#include "std_msgs/Float64MultiArray.h"
#include "nav_msgs/Odometry.h"
#include "tf/tf.h"

#include <torch//script.h>

#include "utils.hpp"
#include "utils_coordinate.h"

#include <deque>

using namespace std;
class Net
{
private:
    torch::jit::script::Module _model;

public:
    int load(const string &modelPath)
    {
        auto ret = isExisting(modelPath);
        if (!ret)
        {
            ROS_ERROR("Model does not exist, please check the model path.");
            return -1;
        }
        ROS_INFO("Model file detected. Loading model...");
        _model = torch::jit::load(modelPath, at::kCUDA);
        ROS_INFO("Model loaded successfully");

        return 0;
    }

    int forward(const at::Tensor &laser, const at::Tensor &goal, const at::Tensor &speed, vector<double> &output)
    {
        if (!laser.is_cuda() or !goal.is_cuda() or !speed.is_cuda())
        {
            ROS_ERROR("[ERROR] Tensors are not in CUDA");
            return -1;
        }

        vector<torch::jit::IValue> inputs = {laser, goal, speed};

        at::Tensor result;
        try
        {
            result = _model.forward(inputs).toTensor().tensor_data();
        }
        catch (exception &e)
        {
            ROS_ERROR("%s", e.what());
            return -1;
        }

        output = {result[0][0].item<double>(), result[0][1].item<double>()};

        return 0;
    }
};

class Env
{
public:
    /***** Variables *****/
    // 四个需要获取的数据
    vector<double> planningVec = {0, 0};
    vector<double> odomSpeed;
    vector<float> rawScanData;
    vector<double> state_GT;

    vector<double> goal_point = {-2.0,3.0}; //stage
    double distanceTolerate = 0.5;

    /***** Functions *****/
    explicit Env(ros::NodeHandle &nh)
    {
        _nh = nh;
        // usual
        _laserScanSub = _nh.subscribe("/robot_7/base_scan", 1, &Env::laserScanCallback, this);
        _speedSub = _nh.subscribe("/robot_7/odom", 1, &Env::odometryCallback, this);

        // real
        _planningVecSub = _nh.subscribe("/navigation_main/planning_vec", 1, &Env::planningVecCallback, this);
        _poseSub = _nh.subscribe("/pose_publisher", 1, &Env::stateCallback, this);  //real

        // unreal
        _stateGTSub = _nh.subscribe("/robot_7/base_pose_ground_truth", 1, &Env::stateGTCallback, this); //stage
        _keySpeedSub = _nh.subscribe("/robot_7/cr_vel", 1, &Env::keyCallback, this);  //key in
        _outputPub = _nh.advertise<geometry_msgs::Twist>("/robot_7/cmd_vel", 1);

        // visualize
        _lidarFilteredPub = _nh.advertise<sensor_msgs::LaserScan>("/lidar_filtered",10);
    }

    void setParams(int input_size, int fov, double max_range, int out_lines,double goal_x, double goal_y, double distanceT)
    {
        _input_size = input_size;
        _fov = fov;
        _max_range = max_range;
        _out_lines = out_lines;
        goal_point = {goal_x,goal_y};
        distanceTolerate = distanceT;
    }

    void output(double linear, double angular)
    {
        geometry_msgs::Twist moveCmd;
        moveCmd.linear.x = linear;
        moveCmd.linear.y = 0;
        moveCmd.linear.z = 0;
        moveCmd.angular.x = 0;
        moveCmd.angular.y = 0;
        moveCmd.angular.z = angular;
        _outputPub.publish(moveCmd);
    }

    void outputKey()
    {
        _outputPub.publish(_keySpeed);
    }

    //TODO：待测试
    at::Tensor get_laser_polar()
    {
        auto lidar_tensor = at::tensor(rawScanData);
        auto max_tensor = at::full_like(lidar_tensor,_max_range);
        lidar_tensor = at::where(at::isnan(lidar_tensor),max_tensor,lidar_tensor);
        lidar_tensor = at::where(at::isinf(lidar_tensor),max_tensor,lidar_tensor);
        lidar_tensor = at::where(lidar_tensor<0.1,max_tensor,lidar_tensor);
        lidar_tensor = at::where(lidar_tensor>_max_range,max_tensor,lidar_tensor);
        auto lidar_polar = u_c::range2polar(lidar_tensor, _fov*M_PI/180);
        return lidar_polar;
    }

    //TODO：待测试
    bool update_his(const vector<double> &xya_his)
    {
        auto xya_now = state_GT;
        auto dis = sqrt(pow(xya_now[0] - xya_his[0], 2) + pow(xya_now[1] - xya_his[1],2));
        auto theta = abs(xya_his[2] - xya_now[2]);
        if (dis>0.2 || theta>0.175)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    //TODO：待测试
    vector<at::Tensor> polars_full2r_filtered_rad(const deque<at::Tensor> &obs_his, const deque<vector<double>> &poses_his)
    {
        auto obs_his_to_now = u_c::hisPolars2nowPolar_rad(obs_his,poses_his,state_GT,_max_range);
        auto lidar_polar = get_laser_polar();
        auto obs_full_polar = at::cat({obs_his_to_now,lidar_polar.unsqueeze(0)},0);
        auto full_r_id = u_c::polars2index(obs_full_polar,_out_lines,2*M_PI);
        auto r_t = u_c::index_filtered(full_r_id, _max_range, _out_lines, _input_size, _fov);
        publishLaserFiltered(r_t);
        return r_t;
    }

    void get_local_goal()
    {
        auto local_x = (goal_point[0] - state_GT[0]) * cos(state_GT[2]) + (goal_point[1] - state_GT[1]) * sin(state_GT[2]);
        auto local_y = -(goal_point[0] - state_GT[0]) * sin(state_GT[2]) + (goal_point[1] - state_GT[1]) * cos(state_GT[2]);
        planningVec = {local_x,local_y};
    }

    bool get_terminate()
    {
        auto distance = sqrt(pow((goal_point[0] - state_GT[0]),2) + pow((goal_point[1] - state_GT[1]),2));
        if(distance<0.5)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

private:

    /***** Variables *****/
    ros::NodeHandle _nh;
    geometry_msgs::Twist _keySpeed;


    /***** Params *****/
    int _input_size;
    int _fov;
    float _max_range;
    int _out_lines;

    /***** Subscribers *****/
    ros::Subscriber _laserScanSub;
    ros::Subscriber _planningVecSub;
    ros::Subscriber _speedSub;
    ros::Subscriber _poseSub;
    ros::Subscriber _stateGTSub;
    ros::Subscriber _keySpeedSub;

    /***** Publishers *****/
    ros::Publisher _outputPub;
    ros::Publisher _lidarFilteredPub;

    /***** Functions *****/
     void laserScanCallback(const sensor_msgs::LaserScan::ConstPtr &msg)
    {
         rawScanData = msg->ranges;
    }

    void planningVecCallback(const std_msgs::Float64MultiArrayConstPtr &msg)
    {
        if ((msg->data).size() != 2)
        {
            ROS_ERROR("The size of planning_vector should be 2");
            planningVec = {0, 0};
        }
        else
            planningVec = msg->data;
    }

    void odometryCallback(const nav_msgs::Odometry::ConstPtr &msg)
    {
        odomSpeed = {msg->twist.twist.linear.x, msg->twist.twist.angular.z};
    }

    //TODO：callback需要看一下转换是否正确
    void stateCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
    {
        tf::Quaternion RQ2;
        double roll,pitch,yaw;
        tf::quaternionMsgToTF(msg->pose.orientation,RQ2);
        tf::Matrix3x3(RQ2).getRPY(roll,pitch,yaw);
        state_GT = {msg->pose.position.x, msg->pose.position.y, yaw};
    }

    void stateGTCallback(const nav_msgs::Odometry::ConstPtr &msg)
    {
        tf::Quaternion RQ2;
        double roll,pitch,yaw;
        tf::quaternionMsgToTF(msg->pose.pose.orientation,RQ2);
        tf::Matrix3x3(RQ2).getRPY(roll,pitch,yaw);
        state_GT = {msg->pose.pose.position.x, msg->pose.pose.position.y, yaw};
    }

    void keyCallback(const geometry_msgs::Twist::ConstPtr &msg)
    {
        _keySpeed.linear.x = msg->linear.x;
        _keySpeed.linear.y = 0;
        _keySpeed.linear.z = 0;

        _keySpeed.angular.x = 0;
        _keySpeed.angular.y = 0;
        _keySpeed.angular.z = msg->angular.z;
    }

    // 发布融合后的雷达数据
    void publishLaserFiltered(const vector<at::Tensor> &r_t)
    {   try
        {
            std::vector<float> r_filtered(r_t[0].data_ptr<double>(), r_t[0].data_ptr<double>() + r_t[0].numel());
            std::vector<float> t_features(r_t[1].data_ptr<float>(), r_t[1].data_ptr<float>() + r_t[1].numel());
            sensor_msgs::LaserScan laser_filtered;
            laser_filtered.ranges = r_filtered;
            laser_filtered.header.frame_id = "/robot_7/base_laser_link";
            laser_filtered.angle_min = 0;
            laser_filtered.angle_max = 2 * M_PI;
            laser_filtered.range_max = _max_range + 1;
            laser_filtered.angle_increment = 2 * M_PI / _out_lines;
            laser_filtered.intensities = t_features;
            _lidarFilteredPub.publish(laser_filtered);
        }
        catch(std::bad_alloc)
        {
            cout<<"bad pub"<<endl;
        }

    }

};

int main(int argc, char **argv)
{
    /***** Initialize *****/
    ros::init(argc, argv, "collision_avoidance", ros::InitOption::AnonymousName);
    ros::NodeHandle pnh("~");

    /***** Parameters *****/
    string modelPath;
    int input_size, fov, out_lines;
    double linearGain, angularGain, max_range;
    bool realFlag;
    double goal_x, goal_y, distanceTolerate;

    pnh.param<string>("model_path", modelPath, "/home/joeray/ros_workspace/ca_360_ws/src/collision_avoidance_360/model/Stage2_360_test.tjm");
    pnh.param<int>("input_size", input_size, 512);
    pnh.param<int>("fov", fov, 180);
    pnh.param<int>("out_lines", out_lines, 1080);
    pnh.param<double>("linear_gain", linearGain, 0.2);
    pnh.param<double>("angular_gain", angularGain, 0.4);
    pnh.param<double>("max_range", max_range, 3.0);
    pnh.param<bool>("real_or_not", realFlag, false);
    pnh.param<double>("goal_x", goal_x, -2.0);
    pnh.param<double>("goal_y", goal_y, 3.0);
    pnh.param<double>("dis_tolerate", distanceTolerate, 0.5);


    /***** Robot Environment *****/
    Env env(pnh);
    env.setParams(input_size, fov, max_range, out_lines, goal_x, goal_y, distanceTolerate);

    /***** Check Ready *****/
    while (env.rawScanData.empty() or env.odomSpeed.empty() or env.state_GT.empty())
    {
        ros::spinOnce();
        ros::Duration(0.5).sleep();
        ROS_WARN("Inputs are not prepared");
    }
    ROS_INFO("Inputs are prepared.Let's get it");

    auto lidar_polar = env.get_laser_polar();
    deque<at::Tensor> obs_his{lidar_polar,lidar_polar,lidar_polar,lidar_polar};
    deque<vector<double>> poses_his{env.state_GT,env.state_GT,env.state_GT,env.state_GT};
    vector<at::Tensor> r_t;
    r_t = env.polars_full2r_filtered_rad(obs_his,poses_his);

    ros::Rate rate(1);
    while (ros::ok())
    {
        ros::spinOnce();

        // 如果是在stage仿真中，需要手动更新planningVec，并且判断是否到达终点
        if(!realFlag)
        {
            if(env.get_terminate())
            {
                ROS_INFO("Reached goal, exit~");
                env.output(0.0, 0.0);
                return 0;
            }

            env.get_local_goal();
        }

        if(env.update_his(poses_his[3]))
        {
            obs_his.pop_front();
            obs_his.emplace_back(env.get_laser_polar());
            poses_his.pop_front();
            poses_his.emplace_back(env.state_GT);
        }

        //act
        env.outputKey();

        // splice
        r_t = env.polars_full2r_filtered_rad(obs_his,poses_his);


//        rate.sleep();
    }
    return 0;
}