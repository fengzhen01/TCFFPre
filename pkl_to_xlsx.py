import pandas as pd
import pickle
# # 加载模型权重
# model_weights = torch.load('E:/Epan/读论文/RPI论文/4.15下载/PPI_GNN-main/Human_features/processed/1A0N.pt')
# model_weights1 = model_weights

# 打开并读取.pkl文件
with open('E:/Epan/multi_channel_PBR/dataset/74_encode_data.pkl','rb') as file:
    content = pickle.load(file)

# # 创建一个空的DataFrame
# all_data = pd.DataFrame()
#
# # 遍历列表中的每个ndarray
# for ndarray in content:
#     # 将ndarray转换为DataFrame
#     df = pd.DataFrame(ndarray)
#     # 将当前DataFrame垂直堆叠到all_data DataFrame中
#     all_data = pd.concat([all_data, df], ignore_index=True)
#
# # 保存合并后的DataFrame到Excel文件中
# excel_path = 'E:/Epan/读论文/RPI论文/4.15下载/PPI_GNN-main/Human_features/combined_data.xlsx'
# all_data.to_excel(excel_path, index=False)  # 设置index=False以防在Excel中保存行索引
#
# print(f"所有数据已成功保存到 {excel_path}")

# 遍历列表中的每个ndarray，每个ndarray是n*1024的形状
for index, ndarray in enumerate(content, start=1):
    # 将ndarray转换为DataFrame
    df = pd.DataFrame(ndarray)

    # 构建保存路径
    excel_path = f'E:/Epan/multi_channel_PBR/dataset/excel_{index}.xlsx'

    # 保存DataFrame到Excel文件
    df.to_excel(excel_path, index=False)  # 设置index=False以防在Excel中保存行索引

    print(f"列表 {index} 的数据已成功保存到 {excel_path}")
