//
// Created by joeray on 2022/2/6.
//

#include <iostream>
#include <torch/torch.h>
#include <torchscatter/scatter.h>
using namespace torch::indexing;
int main() {
    torch::Tensor src = torch::tensor({0.5, 0.4, 0.1, 0.6});
    torch::Tensor index = torch::tensor({0, 0, 1, 1});
    std::cout << src << std::endl;
    std::cout << index << std::endl;
    std::cout << scatter_sum(src, index, 0, torch::nullopt, torch::nullopt) << std::endl;
    at::Tensor a,b;
    auto c = scatter_min(src,index,0,torch::nullopt,torch::nullopt);
    std::tie(a,b) = c;
    std::cout<<"a: "<<a<<std::endl;
    std::cout<<"b: "<<b<<std::endl;
    auto id = at::tensor({0,2,3,1}).toType(at::kLong);
    auto d = src.index({id});
    std::cout<<"d: "<<d<<std::endl;
//    std::cout<<torch::where(src>0.3).size(); //1
    d = at::where(d>0.1,d.toType(at::kDouble),1.0);
    std::cout<<d<<std::endl;
    std::cout<<d.index({Slice(3,None)})<<std::endl;
    std::vector<double> v(d.data_ptr<double>(),d.data_ptr<double>()+d.numel());
    std::cout<<"v: "<<v<<std::endl;
}