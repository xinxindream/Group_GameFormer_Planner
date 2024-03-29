# Group_GameFormer_Planner
## 一、简介
1. 主体为GameFormer流程跑通
2. 处理我们自己的数据符合GameFormer_Planner所需格式，跑通流程，所以不会修改模型代码

## 二、环境安装
1. 下载nuplan数据集
2. 安装[nuPlan devkit](https://nuplan-devkit.readthedocs.io/en/latest/installation.html)(version tested: v1.2.2).
3. 下载源码为nuplan虚拟空间安装必要依赖
```shell
conda activate nuplan

pip install -r requirements.txt
```

## 三、操作流程（GameFormer_Planner）
### 1、相关文件
```shell
# 数据处理
data_process.py

# 模型训练
train_predictor.py

# 测试
run_nuplan_test.py

# 独立可视化测试结果
run_nuboard.py
```

### 2、数据处理
```shell
python data_process.py
--data_path /data/datasets/nuplan/splits/mini
--map_path /data/datasets/nuplan/maps
--save_path /data/datasets/nuplan/processed_data
```

> tips: 官方的data_process.py中是没有进行train_set和valid_set分割的，需要我们自己去分割，我这里在data_process中进行了处理  
> 结果保存：/data/datasets/nuplan/processed_data/train /data/datasets/nuplan/processed_data/valid

### 3、模型训练
```shell
python train_predictor.py 
--train_set /data/datasets/nuplan/processed_data/train 
--valid_set /data/datasets/nuplan/processed_data/valid
```

> tips: 就是每一轮的模型都保存了，后面有评价数值，选最小的就行  
> 模型最终保存位置：./training_log/{experiment_name}/

### 4、测试
```shell
python run_nuplan_test.py
--experiment_name open_loop_boxes
--data_path /data/datasets/nuplan/splits/mini
--map_path /data/datasets/nuplan/maps
--model_path training_log/your/model
```

> tips: 测试时间很长，所以为了可视化之前的测试结果，我们编写了run_nuboard.py可以进去看看怎么改  
> 测试结果保存位置：./testing_log/{experiment_name}/gameformer_planner/{experiment_time}

### 5、可视化
1. 可视化测试结果
> python run_nuboard.py

## 四、操作流程（Group_GameFormer_Planner）
### 1、相关文件
```shell
# 数据处理
xiaoba_rosbag_tonuplan_likenuplan.py

# 模型训练
train_predictor.py

# 测试
run_nuplan_test.py

# 独立可视化测试结果
run_nuboard.py
```

### 2、数据处理
```shell
python xiaoba_rosbag_tonuplan_likenuplan.py
```

> tips: 需要修改该文件内的rosbag、save_path、self._original_route_lane_data_x_path、self._shifit_route_lane_data_x_path，当然还有y轴的路径  
> 处理好的训练集：/data/datasets/xiaoba/2024.1.11/2024-01-11-17-20-37_part2_with_det_2_train/train/   
> 处理好的验证集：/data/datasets/xiaoba/2024.1.11/2024-01-11-17-20-37_part2_with_det_2_train/valid/

### 3、模型训练
```shell
python train_predictor.py 
--train_set /data/datasets/xiaoba/2024.1.11/2024-01-11-17-20-37_part2_with_det_2_train/train/  
--valid_set /data/datasets/xiaoba/2024.1.11/2024-01-11-17-20-37_part2_with_det_2_train/valid/
```

> tips: 就是每一轮的模型都保存了，后面有评价数值，选最小的就行  
> 模型最终保存位置：./training_log/{experiment_name}/

### 4、测试
#### 4.1 GameFormerPlanner测试
```shell
python run_nuplan_test.py
--experiment_name open_loop_boxes
--data_path /data/datasets/nuplan/splits/mini
--map_path /data/datasets/nuplan/maps
--model_path training_log/your/model
```

> tips: 测试时间很长，所以为了可视化之前的测试结果，我们编写了run_nuboard.py可以进去看看怎么改  
> 测试结果保存位置：./testing_log/{experiment_name}/gameformer_planner/{experiment_time}

#### 4.2 Rviz可视化rosbag测试
```shell
# 先启动roscore
roscore

# 执行测试文件
python xiaoba_rosbag_test-likenuplan.py  --model_path /home/user/workspace/pxf/GameFormer-Planner/training_log/Exp3_likenuplan/model_epoch_20_valADE_0.4715.pth

# 测试进行，启动rviz
rviz -d xiaoba_rosbag_test.rviz 
```

> tips: 一定要启动roscore再执行测试文件，不然会因为缺少依赖，测试运行失败  
> rviz过早打开会没有画面的，不用担心  
> 需要修改xiaoba_rosbag_test-likenuplan.py

### 5、可视化
1. 可视化测试结果
> python run_nuboard.py