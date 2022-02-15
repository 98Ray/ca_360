# 使用说明
## 1.运行
### 1.1 指令
`roslaunch collision_avoidance_360 stage_test.launch`
### 1.2 环境
在实车与stage仿真中均可完成测试，但需要参照以下说明修改对应话题

## 2.相关说明
### 2.1 参数说明
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

### 2.2 话题说明
`~/scan`:原始雷达话题   
`~/odom`:提供机器人精确速度(仿真)或计算出的估计速度(实际)   
`~/planning_vec`:提供目标点在机器人坐标系下的局部坐标vector   
`~/pose_publisher`:提供机器人的计算出的估计位姿(实际)     
`~/base_pose_ground_truth`:提供机器人的精确位姿(仿真)     
`~/key_vel`:用于在stage仿真中接收键盘指令   
`~/output_vel`:最终输出控制速度的话题