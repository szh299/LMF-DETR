# LMF-DETR
 A Feature Enhancement Algorithm for Steel Surface Defect Detection
### 训练命令

##### nohup<主要用于linux后台训练> 必看视频-深度学习炼丹小技巧:https://www.bilibili.com/video/BV1q3SZYsExc/
nohup xxx > logs/xxx.log 2>&1 & tail -f logs/xxx.log
###### 示例
CUDA_VISIBLE_DEVICES=0 nohup python train.py -c configs/deim/deim_hgnetv2_n_custom.yml --seed=0 > train.log 2>&1 & tail -f train.log  
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.run --nproc_per_node 4 train.py -c configs/deim/deim_hgnetv2_n_custom.yml --seed=0 > train.log 2>&1 & tail -f train.log

##### 普通训练命令

单卡用法： CUDA_VISIBLE_DEVICES=<显卡id> python train.py -c <yml的路径> --seed=0  
单卡例子： CUDA_VISIBLE_DEVICES=0 python train.py -c configs/deim/deim_hgnetv2_n_custom.yml --seed=0  

多卡用法： CUDA_VISIBLE_DEVICES=<显卡id> python -m torch.distributed.run --nproc_per_node <选用的显卡数量> train.py -c <yml的路径> --seed=0  
多卡例子： CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node 4 train.py -c configs/deim/deim_hgnetv2_n_custom.yml --seed=0  
Windows多卡是不支持CUDA_VISIBLE_DEVICES=0,1,2,3指定的，要换一种方法，具体在train.py顶部有标注

### 测试命令

说明： test-only状态下会输出FPS、模型权重大小、计算量、参数量指标、TIDE指标、保存预测结果的coco-json文件。
用法： python train.py -c <yml的路径> --test-only -r <权重的路径> ｜ 请注意，yml和权重的结构必须一致，不然会报载入失败的问题  
例子： python train.py -c configs/test/dfine_hgnetv2_n_visdrone.yml --test-only -r /home/waas/best_stg2.pth  

### 推理命令(字体和框的大小请看tools/inference/<torch_inf.py,onnx_inf.py,trt_inf.py>的draw函数注释)

##### torch模型推理命令
用法： python tools/inference/torch_inf.py -c <yml的路径> -r <权重的路径> --input <需要检测的路径，支持单张图片、单个视频、一个文件夹>  --output <保存路径> -t <置信度,默认为0.2>  
例子： python tools/inference/torch_inf.py -c configs/dfine/dfine_hgnetv2_n_custom.yml -r /home/waas/best_stg2.pth --input image.png --output inference_results/exp -t 0.2

##### onnx模型推理命令
用法： python tools/inference/onnx_inf.py -p <onnx权重的路径> --input <需要检测的路径，支持单张图片、单个视频、一个文件夹>  --output <保存路径> -t <置信度,默认为0.2>  
例子： python tools/inference/onnx_inf.py -p model.onnx --input image.png --output inference_results/exp -t 0.2 

##### tensorrt模型推理命令
用法： python tools/inference/trt_inf.py -p <tensorrt权重的路径> --input <需要检测的路径，支持单张图片、单个视频、一个文件夹>  --output <保存路径> -t <置信度,默认为0.2>  
例子： python tools/inference/trt_inf.py -p model.engine --input image.png --output inference_results/exp -t 0.2 

### 计算yml的参数量和计算量功能
用法： python tools/benchmark/get_info.py -c <yml的路径>  
例子： python tools/benchmark/get_info.py -c configs/dfine/dfine_hgnetv2_n_custom.yml

### 输出yml的全部参数
用法： python show_yml_param.py -c <yml的路径>   
例子： python show_yml_param.py -c configs/dfine/dfine_hgnetv2_n_custom.yml
