import warnings
warnings.filterwarnings("ignore")

from common_utils import *

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.nuboard.nuboard import NuBoard

def run_nuboard(experiment_name, experiment_time, data_root, map_root):
    """
        function: 开启nuBoard, 可视化测试结果
        parameters: 
            1. experiment_name:
                default='open_loop_boxes',
                'closed_loop_nonreactive_agents',
                'closed_loop_reactive_agents'
            2. experiment_time: 查看testing_log下所需要导入的测试结果的时间
            3. data_root: nuplan数据保存路径
            4. map_root: nuplan里的地图保存路径
        tips:
            实际上就是读取.nuboard所在路径, 其实进入nuboard后还可以继续上传文件
    """
    output_dir = f"testing_log/{experiment_name}/gameformer_planner/{experiment_time}"

    # 建设地图场景
    print('Extracting scenarios...')
    sensor_root = None
    db_files = None
    map_version = "nuplan-maps-v1.0"
    scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5)
    builder = NuPlanScenarioBuilder(data_root, map_root, sensor_root, db_files, map_version, scenario_mapping=scenario_mapping)
    del scenario_mapping

    # 获取output_dir目录下的.nuboard文件列表，后续加载进nuBoard，不包含其子目录里的.nuboard文件
    # simulation_file = 'testing_log/{experiment_name}/gameformer_planner/{experiment_time}/xxx.nuboard'
    simulation_file = [str(file) for file in pathlib.Path(output_dir).iterdir() if file.is_file() and file.suffix == '.nuboard']
    
    nuboard = NuBoard(
        nuboard_paths=simulation_file,
        scenario_builder=builder,
        vehicle_parameters=get_pacifica_parameters()
    )
    nuboard.run()


if __name__ == "__main__":
    experiment_name = 'open_loop_boxes'
    experiment_time = '2024-03-06 15:40:29.079113'
    data_root = '/data/datasets/nuplan/splits/mini'
    map_root = '/data/datasets/nuplan/maps'
    run_nuboard(experiment_name, experiment_time, data_root, map_root)
    

