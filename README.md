# Group_GameFormer_Planner
## 一、简介
1. 主体为GameFormer流程跑通
2. 处理我们自己的数据符合GameFormer_Planner所需格式，跑通流程，所以不会修改模型代码

## 二、环境安装
因为环境依赖什么的，不是我配置的，所以忽略这一过程

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
--data_path /data/dataset/nuplan/splits/mini
--map_path /data/dataset/nuplan/maps
--save_path /data/dataset/nuplan/try_processed_data
```

> tips: 官方的data_process.py中是没有进行train_set和valid_set分割的，需要我们自己去分割，我这里在data_process中进行了处理  
> 结果保存：/data/dataset/nuplan/try_processed_data/train /data/dataset/nuplan/try_processed_data/valid

### 3、模型训练
```shell
python train_predictor.py 
--train_set /data/dataset/nuplan/try_processed_data/train 
--valid_set /data/dataset/nuplan/try_processed_data/valid
```

> tips: 就是每一轮的模型都保存了，后面有评价数值，选最小的就行  
> 模型最终保存位置：./training_log/{experiment_name}/

### 4、测试
```shell
python run_nuplan_test.py
--experiment_name open_loop_boxes
--data_path /data/dataset/nuplan/splits/mini
--map_path /data/dataset/nuplan/maps
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


# 模型训练
train_predictor.py

# 测试
run_nuplan_test.py

# 独立可视化测试结果
run_nuboard.py
```

### 2、数据处理
```shell

```

### 3、模型训练
```shell

```

### 4、测试
```shell

```