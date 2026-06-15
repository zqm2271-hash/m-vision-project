import os

dataset_path = r"D:\lyxxx\3\data\15-Scene"

for folder in sorted(os.listdir(dataset_path)):
    
    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):

        files = os.listdir(folder_path)

        print(folder)
        print("图片数量:", len(files))

        print("前5个文件:")
        print(files[:5])

        print("-"*30)