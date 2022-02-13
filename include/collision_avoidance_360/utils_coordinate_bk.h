//
// Created by joeray on 2022/1/31.
//

#ifndef SRC_UTILS_COORDINATE_H
#define SRC_UTILS_COORDINATE_H

#include <iostream>
#include <torch/script.h>
#include <torchscatter/scatter.h>
#include <vector>
#include <deque>
#include <math.h>
#include <exception>

using namespace std;
using namespace torch::indexing;

namespace u_c
{

    at::Tensor polar2cart_rad(const at::Tensor &input_rp)
    {
    //    cos = torch.cos(input_rp[:, 1])
    //    sin = torch.sin(input_rp[:, 1])
    //    x = torch.mul(input_rp[:, 0], cos)
    //    y = torch.mul(input_rp[:, 0], sin)
    //    return torch.stack((x, y), dim=1)
        auto cos = at::cos(input_rp.select(1,1));
        auto sin = at::sin(input_rp.select(1,1));
        auto x = at::mul(input_rp.select(1,0),cos);
        auto y = at::mul(input_rp.select(1,0),sin);
        return at::stack({x,y},1);
    }

    at::Tensor cart2polar(const at::Tensor &input_xy, const double max_range)
    {
        auto r = at::sqrt(at::pow(input_xy.select(1,0),2) + at::pow(input_xy.select(1,1),2));
        r.clamp_max_(max_range);
        auto p = at::atan2(input_xy.select(1,1),input_xy.select(1,0));
        return at::stack({r,p},1);
    }

    at::Tensor range2polar(const at::Tensor &ranges, const double &fov)
    {
        auto num_lines = ranges.numel();
        auto resolution = fov/num_lines;
        auto ranges_theta = at::arange(-fov/2.0+resolution/2.0, fov/2.0, resolution);
        auto ranges_polar = at::stack({ranges,ranges_theta},1);
        return ranges_polar;
    }

    at::Tensor pose2matrix(const vector<double> &pose)
    {
        auto theta_rad = pose[2];
        auto cos_theta = cos(theta_rad);
        auto sin_theta = sin(theta_rad);
        auto t_x = pose[0];
        auto t_y = pose[1];
        auto matrix = at::zeros({3,3});
        matrix[0][0] = cos_theta;
        matrix[0][1] = -sin_theta;
        matrix[0][2] = t_x;
        matrix[1][0] = sin_theta;
        matrix[1][1] = cos_theta;
        matrix[1][2] = t_y;
        matrix[2][2] = 1;
        return matrix;
    }

    at::Tensor his_xy2now_xy(const vector<double> &xya_his, const vector<double> &xya_now, const at::Tensor &xy_his)
    {
        auto his_m = pose2matrix(xya_his);
        auto now_m = pose2matrix(xya_now);;
        auto xy_his_homo = at::cat({xy_his.transpose(0,1),at::ones({1,xy_his.size(0)})},0);
        auto tensor_1 = at::mm(now_m.inverse(),his_m).toType(at::kDouble);
        auto tensor_2 = xy_his_homo.toType(at::kDouble);
        auto xy_now = at::mm(tensor_1,tensor_2).slice(0,0,2).transpose(0,1);
        return xy_now;
    }

    at::Tensor hisPolar2nowPolar_rad(const at::Tensor &rp_his, const vector<double> &xya_his, const vector<double> &xya_now, const double &max_range)
    {
        auto xy_his = polar2cart_rad(rp_his);
        auto xy_now = his_xy2now_xy(xya_his, xya_now, xy_his);
        auto rp_now = cart2polar(xy_now, max_range);
//      因为雷达未扫到的点被初始化为max_range
//      然后当机器人向前移动且依然没有被遮挡时，会出现未被遮挡处的点逐渐靠近的情况
//      以下的操作是为了将初始化为max_range（也就是补的max_range值）不做变换
//      以此来避免上述情况的出现
        auto r_tmp = at::where(rp_his.select(1,0)==max_range,rp_his.select(1,0),rp_now.select(1,0).toType(at::kFloat));
        rp_now.transpose(0,1)[0] = r_tmp;
        return rp_now;
    }

    at::Tensor hisPolars2nowPolar_rad(const deque<at::Tensor> &rps_his, const deque<vector<double>> &xyas_his, const vector<double> &xya_now, const double &max_range)
    {
        vector<at::Tensor> rps_now;
        for(int i=0;i<rps_his.size();i++)
        {
            rps_now.emplace_back(hisPolar2nowPolar_rad(rps_his[i],xyas_his[i],xya_now,max_range));
        }
        return at::stack(rps_now,0);
    }

    at::Tensor polar2index(const at::Tensor &input_rp, const int &out_lines, const double &scope_keep=M_PI*2)
    {
        input_rp.select(1,1) = at::where(input_rp.select(1,1)<0,input_rp.select(1,1)+scope_keep,input_rp.select(1,1));
        auto step = scope_keep/out_lines;
        auto output_r = input_rp.select(1,0);
        auto output_id = (input_rp.index_select(1,at::tensor({1}))/step).toType(at::kInt);
        auto output_r_id = at::stack({output_r,output_id.squeeze(1)},1);
        return output_r_id;
    }

    at::Tensor polars2index(const at::Tensor &input_rps, const int &out_lines, const double &scope_keep=M_PI*2)
    {
        vector<at::Tensor> output_r_ids;
        for(int i=0;i<input_rps.size(0);i++)
        {
            output_r_ids.emplace_back(polar2index(input_rps[i],out_lines,scope_keep));
        }
        return at::stack(output_r_ids);
    }

    vector<at::Tensor> index_filtered(const at::Tensor &n_r_id, const double &max_range, const int &out_lines)
    {
        // n_r_id是his*N*2的tensor，也就是要把所有观测放在一个tensor中再送进来；max_range是雷达的最大值；out_lines是最终输出的线数
        auto tmp = at::transpose(n_r_id,1,2);
        vector<at::Tensor> new_r_id_vec;
        // 把所有的r拼成一行，id拼成一行
        // 最终拼完的new_r_id 形状：2*(his*N),一行是r，一行是id，每行有his*N个数据
        for(int i=0;i<tmp.size(0);i++)
        {
            new_r_id_vec.emplace_back(tmp[i]);
        }
        auto new_r_id = at::cat(new_r_id_vec,1);

        // 关于时间的一维特征信息，因为从前到后是t越来越靠近当前，所以时间特征越来越大，以0.2倍增
        // TODO:以下一部分操作和输入的雷达线数有关，如果不是512的话需要改一下
        auto t_feature = at::empty(2560);
        for(int i=0;i<5;i++)
        {
            t_feature.slice(0,i*512,(i+1)*512) = 0.2*(i+1);
        }

        // 返回的均为OUT_LINES长度的tensor
        // 获取到对应角度的最小值以及最小值对应的序号
        at::Tensor filtered_r, argmin;
        std::tie(filtered_r, argmin) = scatter_min(new_r_id[0],new_r_id[1].toType(at::kLong),0,torch::nullopt,torch::nullopt);
        //TODO:输出对不对
        argmin = at::where(argmin==2560,2559,argmin);
        auto t_feature_new = t_feature.index({argmin.toType(at::kLong)});
        auto id_unknown = at::where(filtered_r == 0);
        //TODO：看一下这个输出对不对
        t_feature_new.index_put_({id_unknown[0]},-1);
        filtered_r = at::where(filtered_r==0,max_range,filtered_r);


        //扩充当前帧的原始数据
        auto r_now = new_r_id[0].index({Slice(4*512,None)}).toType(at::kDouble);
        std::vector<double> src(r_now.data_ptr<double>(),r_now.data_ptr<double>()+r_now.numel());
        auto srcLen = r_now.numel();
        //扩充到1080的一半，为了让眼前的这180度都是确定的当前帧（如果改成150，那么需要把1/2改成150/360）
        auto dstLen = out_lines  / 2;
        auto scale = srcLen * 1.0 / dstLen;
        vector<float> dst(dstLen);
        for (auto dstX = 0; dstX < dstLen; dstX++)
        {
            auto x = (float(dstX) + 0.5) * scale - 0.5;
            auto x1 = int(floor(x));
            auto x2 = int(min(x1 + 1, int(srcLen - 1)));
            if (x1 != x2)
                dst[dstX] = ((x2 - x) / (x2 - x1)) * src[x1] + ((x - x1) / (x2 - x1)) * src[x2];
            else
                dst[dstX] = src[x2];
        }
        auto scan_sparse = at::tensor(dst);
//        cout<<"scan_sparse.sizes()"<<endl<<scan_sparse.sizes()<<endl; //540

        // 用当前帧替换掉filtered_r中的当前观测部分
        int id_near2zero = out_lines*180/360/2;
        int id_near2last = out_lines - out_lines*180/360/2;
        filtered_r.index({Slice(None,id_near2zero)}) = scan_sparse.index({Slice(int(out_lines/4),None)});
        filtered_r.index({Slice(id_near2last,None)}) = scan_sparse.index({Slice(None,int(out_lines/4))});
//        cout<<"filtered_r"<<endl<<filtered_r<<endl;
        // 将当前帧对应的时间特征全部改为1.0
        t_feature_new.index({Slice(None,id_near2zero)}) = 1.0;
        t_feature_new.index({Slice(id_near2last,None)}) = 1.0;
//        cout<<"t_feature_new"<<endl<<t_feature_new<<endl;
        vector<at::Tensor> r_t = {filtered_r,t_feature_new};
        return r_t;
    }

}


#endif //SRC_UTILS_COORDINATE_H
