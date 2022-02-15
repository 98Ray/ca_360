# 使用说明
## 1.环境配置与编译
### 1.1 环境
#### 1.1.1 cuda&cudnn
首先准备好cuda和对应的cudnn库     
作为参照：本人使用过11.1和11.3版本的cuda，其他版本未测试
#### 1.1.2 torch_scatter
1. 首先在torch_scatter的cmakelist中设置Torch_DIR  
`set(Torch_DIR /.../libtorch/share/cmake/Torch)`
2. 其余参照torch_scatter的[github仓库](https://github.com/rusty1s/pytorch_scatter "github仓库")中说明编译即可。


### 1.2 编译代码中存在的一些问题如下：
#### 1.2.1 编译时找不到Python库是因为cmake版本需要在3.12以上，而ubuntu18自带的cmake版本为3.10
升级cmake版本到3.19参照网址：
https://blog.csdn.net/weixin_35757704/article/details/112557853
#### 1.2.2 torch路径找不到
需要在torch_scatter和自己的ros包的cmakelist中设置Torch_DIR
set(Torch_DIR /.../libtorch/share/cmake/Torch)
#### 1.2.3 python.h头文件找不到
需要在cmakelist中国设置include_directories
加入/usr/bin/python3.6m，也就是python.h所在的文件夹，这样编译就能通过了
#### 1.2.4 运行时提示找不到libtorchscatter.so库
查找发现，**这样的问题主要是因为：**在linux执行连接了.so库的可执行程序时，如果未将so文件放入系统指定的so文件搜索路径下，运行程序时会提示找不到对应的so库，输出上方的错误信息，而执行搜索这一工作的是一个叫ld链接器的东西。Linux中的ld链接器找不到所需的该库文件导致报BUG。
Linux链接器ld默认的搜索目录是`/lib`和`/usr/lib`，所以so文件应该放在这些路径下，如果so文件放在其他路径也可以，但是需要修改一些系统文件让ld知道库文件在哪里。
所以将以下命令放在.bashrc中即可：
`export LD_LIBRARY_PATH=/xxxx/zyccc/libtorch/lib:$LD_LIBRARY_PATH`
`export LIBRARY_PATH=/xxxx/zyccc/libtorch/lib:$LIBRARY_PATH`

## 2.运行
### 2.1 指令
`roslaunch collision_avoidance_360 stage_test.launch`
### 2.2 环境
在实车与stage仿真中均可完成测试，但需要参照以下说明修改对应话题

## 3.相关说明
### 3.1 参数说明
`model_path`:torch模型所在位置        
`input_size`:输入的雷达数据长度      
`fov`:雷达fov，单位是°        
`out_lines`:对原始雷达补全后输出的线数       
`linear_gain`:线速度增益     
`angular_gain`:角速度增益        
`max_range`:最大距离，可修改用于改变机器人避障策略的谨慎程度。越大机器人越谨慎       
`real_or_not`:是否是真实环境。真实为true，否则为false      
`goal_x`:用于stage仿真环境,设置目标点的x坐标。真机测试不需要。     
`goal_y`:用于stage仿真环境,设置目标点的y坐标。真机测试不需要。     
`dis_tolerate`:用于stage仿真环境，设置是否到达目标点的距离判定，单位为m。     

### 3.2 话题说明
`~/scan`:原始雷达话题   
`~/odom`:提供机器人精确速度(仿真)或计算出的估计速度(实际)   
`~/planning_vec`:提供目标点在机器人坐标系下的局部坐标vector   
`~/pose_publisher`:提供机器人的计算出的估计位姿(实际)     
`~/base_pose_ground_truth`:提供机器人的精确位姿(仿真)     
`~/key_vel`:用于在stage仿真中接收键盘指令   
`~/output_vel`:最终输出控制速度的话题