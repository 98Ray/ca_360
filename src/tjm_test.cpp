//
// Created by joeray on 2022/1/30.
//

#include <iostream>
#include "ros/ros.h"
#include <torch/script.h>
#include "utils.hpp"

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

int main(int argc, char **argv)
{
    /***** Initialize *****/
    ros::init(argc, argv, "collision_avoidance", ros::InitOption::AnonymousName);
    ros::NodeHandle pnh("~");

    string modelPath;
    pnh.param<string>("model_path", modelPath, "/home/joeray/ros_workspace/ca_360_ws/src/collision_avoidance_360/model/Stage2_360_test.tjm");

    /***** Neural Network *****/
    Net net;
    auto ret = net.load(modelPath);
    if (ret != 0)
    {
        ROS_ERROR("Node [%s] is shutdown", ros::this_node::getName().c_str());
        ros::shutdown();
        return ret;
    }
    else
    {
        cout<< "ret:" << ret <<endl;
    }

    auto tensorOptions = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
    auto laserTensor = at::ones({1,2,1080},tensorOptions);
    auto goalTensor = at::tensor({2.0,0.0},tensorOptions).reshape({-1,2});
    auto speedTensor = at::tensor({0.5,0.0},tensorOptions).reshape({-1,2});

    vector<double> action;
    ret = net.forward(laserTensor, goalTensor, speedTensor, action);
    if (ret != 0)
    {
        ROS_ERROR("Node [%s] is shutdown", ros::this_node::getName().c_str());
        ros::shutdown();
    }
    cout<<action[0]<<","<<action[1]<<endl;

//    cout<<goalTensor.sizes()<<endl;
//    cout<<goalTensor[0]<<endl<<goalTensor[1]<<endl;
//    cout<<speedTensor[0]<<endl<<speedTensor[1]<<endl;
//    auto a = goalTensor[0].item();
//    auto b = goalTensor[1].item();
//    printf("%d,%d",a,b);
    return 0;
}