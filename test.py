import os

objects_list = []

def get_objects_list():
    ''' Get the list of objects '''
    # 获取当前目录下所有xml文件
    dir_1 = "tactile_envs/assets/insertion/complex_objects/continue_1d"
    dir_2 = "tactile_envs/assets/insertion/complex_objects/discrete_2d"
    xml_files_1 = [f for f in os.listdir(dir_1) if f.endswith('.xml')]
    xml_files_2 = [f for f in os.listdir(dir_2) if f.endswith('.xml')]
    xml_files = xml_files_1 + xml_files_2
    
    # 去掉.xml后缀,添加到objects_list
    for xml_file in xml_files:
        obj_name = xml_file[:-4]  # 去掉.xml后缀
        if obj_name not in objects_list:
            objects_list.append(obj_name)
    return objects_list

print(get_objects_list())