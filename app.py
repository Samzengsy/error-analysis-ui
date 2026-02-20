import pandas as pd
import numpy as np
from nicegui import ui
import io
from PIL import Image
import base64
import os
import argparse

class ParquetViewer:
    def __init__(self, parquet_path):
        self.parquet_path = parquet_path
        self.df = None
        self.filtered_df = None
        self.current_index = 0
        self.selected_task_type = "全部"
        self.task_types_list = []
        self.load_data()
        
    def load_data(self):
        """加载parquet文件"""
        try:
            self.df = pd.read_parquet(self.parquet_path)
            print(f"成功加载数据，共 {len(self.df)} 条记录")
            self.extract_task_types()
            self.filter_data()
        except Exception as e:
            print(f"加载parquet文件失败: {e}")
            self.df = pd.DataFrame()
    
    def extract_task_types(self):
        """提取所有任务类型"""
        all_task_types = set()
        for task_types in self.df['task_types']:
            if isinstance(task_types, list):
                all_task_types.update(task_types)
            elif task_types is not None:
                all_task_types.add(str(task_types))
        
        self.task_types_list = ["全部"] + sorted(list(all_task_types))
        print(f"发现任务类型: {self.task_types_list}")
    
    def filter_data(self):
        """根据选择的任务类型筛选数据"""
        if self.selected_task_type == "全部":
            self.filtered_df = self.df.copy()
        else:
            # 筛选包含指定任务类型的记录
            mask = self.df['task_types'].apply(
                lambda x: self.selected_task_type in x if isinstance(x, list) 
                else str(x) == self.selected_task_type if x is not None 
                else False
            )
            self.filtered_df = self.df[mask].copy()
        
        # 重置索引
        self.filtered_df.reset_index(drop=True, inplace=True)
        self.current_index = 0
        print(f"筛选后数据: {len(self.filtered_df)} 条记录")
    
    def on_task_type_change(self, value):
        """任务类型选择变化时的回调"""
        self.selected_task_type = value
        self.filter_data()
        if hasattr(self, 'index_label'):
            self.display_record()
    
    def get_image_from_blob(self, blob_data):
        """从二进制数据转换为图片"""
        try:
            if blob_data is None:
                return None
            
            # 如果是numpy数组，需要特殊处理
            if isinstance(blob_data, np.ndarray):
                # 如果是对象数组，提取实际的图片数据
                if blob_data.dtype == 'O':  # 对象类型
                    # 尝试提取数组中的第一个元素
                    if blob_data.size > 0:
                        actual_data = blob_data.flat[0]  # 获取第一个元素
                        if actual_data is not None:
                            return self.get_image_from_blob(actual_data)  # 递归处理
                    return None
                
                # 如果是字节数据
                try:
                    # 直接尝试作为图片字节流
                    image = Image.open(io.BytesIO(blob_data.tobytes()))
                except:
                    try:
                        # 尝试直接使用数组数据
                        image = Image.open(io.BytesIO(blob_data))
                    except:
                        print(f"无法处理numpy数组，形状: {blob_data.shape}, 类型: {blob_data.dtype}")
                        return None
            
            # 如果是字节串
            elif isinstance(blob_data, (bytes, bytearray)):
                image = Image.open(io.BytesIO(blob_data))
            
            # 如果是其他类型，尝试转换
            else:
                try:
                    if hasattr(blob_data, 'tobytes'):
                        blob_data = blob_data.tobytes()
                    elif hasattr(blob_data, 'read'):  # 文件类对象
                        blob_data = blob_data.read()
                    image = Image.open(io.BytesIO(blob_data))
                except:
                    print(f"无法处理数据类型: {type(blob_data)}")
                    return None
            
            # 转换为base64用于web显示
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            print(f"图片转换失败: {e}")
            return None
    def load_image_from_path(self, image_paths):
        """从文件路径加载图片"""
        try:
            if image_paths is None:
                return None
                
            # 处理路径列表
            if isinstance(image_paths, (list, np.ndarray)):
                if len(image_paths) > 0:
                    path = image_paths[0]
                else:
                    return None
            else:
                path = image_paths
            
            # 检查文件是否存在
            import os
            if not os.path.exists(path):
                print(f"图片文件不存在: {path}")
                return None
            
            # 加载图片
            with open(path, 'rb') as f:
                image_data = f.read()
            
            image = Image.open(io.BytesIO(image_data))
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            print(f"从路径加载图片失败: {e}")
            return None
    
    def create_ui(self):
        """创建用户界面"""
        if self.df.empty:
            ui.label("无法加载数据或数据为空").classes('text-red-500 text-xl')
            return
        
        # 添加键盘事件监听
        ui.keyboard(self.handle_key, active=True)
        
        # 页面标题
        with ui.row().classes('w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white p-4 mb-4 justify-between items-center'):
            ui.label('错误样例分析器').classes('text-3xl font-bold')
            ui.label('SmolVLM2 测试结果').classes('text-lg opacity-90')
        
        # 任务类型选择器
        with ui.card().classes('w-full mb-4 p-4'):
            with ui.row().classes('w-full items-center gap-4'):
                ui.label('任务类型筛选:').classes('text-lg font-semibold')
                self.task_select = ui.select(
                    options=self.task_types_list,
                    value=self.selected_task_type,
                    on_change=lambda e: self.on_task_type_change(e.value)
                ).classes('flex-grow')
                
                # 显示当前筛选结果统计
                self.filter_stats = ui.label().classes('text-sm text-gray-600')
        
        # 主容器
        with ui.row().classes('w-full h-screen gap-4 p-4'):
            # 左侧图片显示区域
            with ui.column().classes('w-1/2'):
                with ui.row().classes('w-full justify-between items-center mb-4'):
                    ui.label('错误样例').classes('text-xl font-bold')
                    self.index_label = ui.label().classes('text-lg font-semibold text-blue-600')
                
                with ui.card().classes('w-full flex-1 p-4'):
                    self.image_container = ui.column().classes('w-full h-full items-center justify-center')
                
                with ui.row().classes('w-full justify-center gap-4 mt-4'):
                    self.prev_btn = ui.button(
                        '← 上一张', 
                        on_click=self.prev_record
                    ).classes('px-6 py-2 bg-blue-500 text-white rounded hover:bg-blue-600')
                    
                    self.next_btn = ui.button(
                        '下一张 →', 
                        on_click=self.next_record
                    ).classes('px-6 py-2 bg-blue-500 text-white rounded hover:bg-blue-600')
                
                ui.label('提示：可使用 ← → 键或 A D 键切换').classes('text-sm text-gray-500 text-center mt-2')
    # 右侧答案显示区域
            with ui.column().classes('w-1/2'):
                ui.label('答案对比').classes('text-xl font-bold mb-4')
                
                with ui.card().classes('w-full mb-4'):
                    ui.label('问题').classes('text-lg font-semibold text-gray-700 mb-2')
                    self.question_display = ui.html().classes('p-3 bg-gray-50 rounded border-l-4 border-gray-400')
                
                with ui.card().classes('w-full mb-4 flex-1'):
                    ui.label('模型答案').classes('text-lg font-semibold text-red-600 mb-2')
                    self.model_answer_display = ui.html().classes('p-3 bg-red-50 rounded border-l-4 border-red-400 h-full overflow-auto')
                
                with ui.card().classes('w-full flex-1'):
                    ui.label('标准答案').classes('text-lg font-semibold text-green-600 mb-2')
                    self.answer_display = ui.html().classes('p-3 bg-green-50 rounded border-l-4 border-green-400 h-full overflow-auto')
                
                with ui.expansion('详细信息', icon='info').classes('w-full mt-4'):
                    with ui.row().classes('w-full gap-4'):
                        with ui.column().classes('flex-1'):
                            self.id_display = ui.label().classes('text-sm')
                            self.conversation_id_display = ui.label().classes('text-sm')
                            self.data_type_display = ui.label().classes('text-sm')
                        
                        with ui.column().classes('flex-1'):
                            self.task_types_display = ui.label().classes('text-sm')
                            self.accuracy_display = ui.label().classes('text-sm')
        
        # 初始显示第一条记录
        self.display_record()
    
    def handle_key(self, e):
        """处理键盘事件"""
        if e.action.keydown:
            if e.key in ['ArrowLeft', 'a', 'A']:
                self.prev_record()
            elif e.key in ['ArrowRight', 'd', 'D']:
                self.next_record()
    
    def display_record(self):
        """显示当前记录"""
        if self.filtered_df.empty or self.current_index >= len(self.filtered_df):
            # 显示无数据状态
            self.image_container.clear()
            with self.image_container:
                ui.label('无匹配的错误样例').classes('text-gray-500 text-xl')
                ui.icon('search_off').classes('text-6xl text-gray-300 mt-4')
            
            self.question_display.content = '<div>无数据</div>'
            self.answer_display.content = '<div>无数据</div>'
            self.model_answer_display.content = '<div>无数据</div>'
            
            self.index_label.text = '0 / 0'
            self.filter_stats.text = f'当前筛选: {self.selected_task_type} (0 条记录)'
            
            self.prev_btn.enabled = False
            self.next_btn.enabled = False
            return
        
        record = self.filtered_df.iloc[self.current_index]
        
        # 显示图片，同时传递路径信息
        self.display_image(
            record.get('images'), 
            record.get('image_paths')
        )
        
        # 显示文本内容
        question = str(record.get('question', '无问题'))
        answer = str(record.get('answer', '无标准答案'))
        model_answer = str(record.get('prediction', '无模型答案'))
        
        # 格式化显示内容
        self.question_display.content = f'<div style="white-space: pre-wrap; line-height: 1.5;">{self.escape_html(question)}</div>'
        self.answer_display.content = f'<div style="white-space: pre-wrap; line-height: 1.5;">{self.escape_html(answer)}</div>'
        self.model_answer_display.content = f'<div style="white-space: pre-wrap; line-height: 1.5;">{self.escape_html(model_answer)}</div>'
        
        # 显示其他信息
        self.id_display.text = f"ID: {record.get('id', 'N/A')}"
        self.conversation_id_display.text = f"对话ID: {record.get('conversation_id', 'N/A')}"
        self.data_type_display.text = f"数据类型: {record.get('data_type', 'N/A')}"
        
        # 处理任务类型
        task_types = record.get('task_types', [])
        if isinstance(task_types, list):
            task_types_str = ', '.join(map(str, task_types))
        else:
            task_types_str = str(task_types)
        self.task_types_display.text = f"任务类型: {task_types_str}"
        # 显示准确率
        accuracy = record.get('accuracy', 'N/A')
        self.accuracy_display.text = f"准确率: {accuracy}"
        
        # 更新索引显示
        self.index_label.text = f'{self.current_index + 1} / {len(self.filtered_df)}'
        
        # 更新筛选统计信息
        total_count = len(self.df)
        filtered_count = len(self.filtered_df)
        self.filter_stats.text = f'当前筛选: {self.selected_task_type} ({filtered_count}/{total_count} 条记录)'
        
        # 更新按钮状态
        self.prev_btn.enabled = self.current_index > 0
        self.next_btn.enabled = self.current_index < len(self.filtered_df) - 1
    
    def escape_html(self, text):
        """转义HTML特殊字符"""
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&#x27;'))
    
    def display_image(self, images_data, image_paths=None):
        """显示单张图片"""
        self.image_container.clear()
        
        with self.image_container:
            # 首先尝试从images数据加载
            img_src = None
            if images_data is not None:
                img_src = self.get_image_from_blob(images_data)
            
            # 如果images数据失败，尝试从路径加载
            if img_src is None and image_paths is not None:
                img_src = self.load_image_from_path(image_paths)
            
            if img_src:
                ui.html(f'''
                    <img src="{img_src}" 
                         style="max-width: 100%; 
                                max-height: 70vh; 
                                height: auto; 
                                border: 2px solid #e5e7eb; 
                                border-radius: 8px; 
                                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
                ''').classes('flex justify-center')
            else:
                ui.label('图片无法显示').classes('text-red-500 text-xl')
                ui.icon('broken_image').classes('text-6xl text-red-300 mt-4')
    
    def prev_record(self):
        """显示上一条记录"""
        if self.current_index > 0:
            self.current_index -= 1
            self.display_record()
    
    def next_record(self):
        """显示下一条记录"""
        if self.current_index < len(self.filtered_df) - 1:
            self.current_index += 1
            self.display_record()

def main():
    parser = argparse.ArgumentParser(description="Visualize error samples from a Parquet file.")
    parser.add_argument(
        "--parquet",
        default=os.environ.get("PARQUET_PATH", "data/error_samples.parquet"),
        help="Parquet file path"
    )
    parser.add_argument("--host", default=os.environ.get("UI_HOST", "127.0.0.1"), help="UI host")
    parser.add_argument("--port", type=int, default=int(os.environ.get("UI_PORT", "8080")), help="UI port")
    args = parser.parse_args()

    ui.page_title = "错误样例查看器"
    viewer = ParquetViewer(args.parquet)
    viewer.create_ui()

    ui.run(
        title="错误样例查看器",
        port=args.port,
        host=args.host,
        reload=False,
        show=False
    )


if __name__ == "__main__":
    main()
