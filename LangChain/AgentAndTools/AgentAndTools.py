import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.tools import Tool

class DataAnalysisTools:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        print(f"数据加载成功！数据形状：{self.df.shape}")
        print(f"数据列：", self.df.columns.to_list())
        print(f"\n前5行数据：")
        print(self.df.head())

    def get_data_summary(self, query):
        """获取数据概览"""
        try:
            summary_info = {
                "数据形状":f"{self.df.shape[0]}行，{self.df.shape[1]}列",
                "数据列":self.df.columns.to_list(),
                "数据类型":self.df.dtypes.astype(str).to_dict(),
                "缺失值统计":self.df.isnull().sum().to_dict(),
                "前3行数据":self.df.head(5).to_dict('records')
            }
            result = "数据概览:\n"
            result += f"- 数据形状：{summary_info['数据形状']}\n"
            result += f"- 数据列：{', '.join(summary_info['数据列'])}\n"
            result += f"- 数据类型：\n"
            for col, dtype in summary_info['数据类型'].items():
                result += f"  - {col}: {dtype}\n"
            result += "- 缺失值:\n"
            for col, missing in summary_info['缺失值统计'].items():
                result += f"  - {col}: {missing}\n"

            return result
        except Exception as e:
            print(f"获取数据概览错误。{e}")
    
    def calculate_statistics(self, query):
        """计算统计信息"""
        try:
            numeric_columns = self.df.select_dtypes(include=['number']).columns
            if len(numeric_columns) == 0:
                return "数据集中没有数值列"
            stats = self.df[numeric_columns].describe()
            result = "数值列统计信息:\n"
            for col in numeric_columns:
                result += f"\n{col}:\n"
                result += f"  计数: {stats[col]['count']:.0f}\n"
                result += f"  均值: {stats[col]['mean']:.2f}\n"
                result += f"  标准差: {stats[col]['std']:.2f}\n"
                result += f"  最小值: {stats[col]['min']:.2f}\n"
                result += f"  25%分位数: {stats[col]['25%']:.2f}\n"
                result += f"  中位数: {stats[col]['50%']:.2f}\n"
                result += f"  75%分位数: {stats[col]['75%']:.2f}\n"
                result += f"  最大值: {stats[col]['max']:.2f}\n"
            
            return result
        except Exception as e:
            print(f"计算统计信息失败。{e}")
    def filter_data(self, query):
        """根据条件过滤数据"""
        try:
            query_lower = query.lower()
            filtered_df = self.df.copy()
            regions = ['north', 'south', 'east', 'west']
            for region in regions:
                if region in query_lower:
                    filtered_df = filtered_df[filtered_df['region'] == region.title()]
                    break
            # 按类别过滤
            categories = ['electronics', 'clothing', 'home']
            for category in categories:
                if category in query_lower:
                    filtered_df = filtered_df[filtered_df['category'] == category.title()]
                    break
            # 按产品过滤
            products = ['product a', 'product b', 'product c', 'product d', 'product e']
            for product in products:
                if product in query_lower:
                    filtered_df = filtered_df[filtered_df['product'] == product.title()]
                    break
            if len(filtered_df) == 0:
                return "没有找到匹配的数据。"
            total_sales = filtered_df['sales'].sum()
            result = f"找到 {len(filtered_df)} 条匹配记录，总销售额：{total_sales}\n"
            result += filtered_df.to_string(index=False)
            return result # 不要掉了这句不要掉了这句不要掉了这句
        except Exception as e:
            print(f"根据条件过滤数据失败。{e}")
            
    def create_visualization(self, query):
        """创建可视化图表"""
        try:
            plt.figure(figsize=(12, 8))
            query_lower = query.lower()
            query_type = "region"
            if any(word in query_lower for word in ['region', '地区']):
                query_type = "region"
                # 按地区统计销售额
                region_sales = self.df.groupby('region')['sales'].sum().sort_values(ascending=False)
                plt.bar(region_sales.index, region_sales.values, color=['#FF6B6B', '#4ECDC4','#45B7D1', '#96CEB4'])
                plt.title('Sales by Region', fontsize=16, fontweight='bold')
                plt.xlabel('Region', fontsize=12)
                plt.ylabel('Total Sales', fontsize=12)
                plt.grid(axis='y', alpha=0.3)
                # 在柱子上添加数值标签
                for i, v in enumerate(region_sales.values):
                    plt.text(i, v + 50, str(v), ha='center', va='bottom')
            elif any(word in query_lower for word in ['category', '类别']):
                query_type = "category"
                category_sales = self.df.groupby('category')['sales'].sum()
                colors = ['#FF6B6B', '#4ECDC4','#45B7D1']
                plt.pie(category_sales.values, labels=category_sales.index, autopct='%1.1f%%', colors=colors,startangle=90)
                plt.title('Sales Distribution by Category', fontsize =16, fontweight='bold')
            elif any(word in query_lower for word in ['daily', 'trend', '趋势']):
                query_type = "trend"
                # 每日销售额趋势
                self.df['date'] = pd.to_datetime(self.df['date'])
                daily_sales = self.df.groupby('date')['sales'].sum()
                plt.plot(daily_sales.index, daily_sales.values, marker='o', linewidth=2, markersize=8, color='#FF6B6B')
                plt.title('Daily Sales Trend', fontsize=16, fontweight='bold')
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Sales', fontsize=12)
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
            
            elif any(word in query_lower for word in ['product', '产品']):
                query_type = "product"
                # 产品销售额
                product_sales = self.df.groupby('product')['sales'].sum().sort_values(ascending=False)
                plt.barh(product_sales.index, product_sales.values, color=['#4ECDC4', '#45B7D1', '#96CEB4', '#FFE66D', '#FF6B6B'])
                plt.title('Sales by Product', fontsize=16, fontweight='bold')
                plt.xlabel('Total Sales', fontsize=12)
            
            else:
                query_type = "default"
                # 默认显示销售额分布
                plt.hist(self.df['sales'], bins=10, edgecolor='black', alpha=0.7, color='#4ECDC4')
                plt.title('Sales Distribution', fontsize=16, fontweight='bold')
                plt.xlabel('Sales', fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'visualization_' + f"{query_type}" + '.png', dpi=300, bbox_inches='tight')
            plt.close()
            return "可视化图表已保存为'visualization.png'"
        except Exception as e:
            return f"创建可视化图表失败。{e}"

    def get_tools(self):
        """返回所有工具"""
        return [
            Tool(
                name = "data_summary",
                func=self.get_data_summary,
                description="获取数据集的概览信息，如均值，标准差，分位数等"
            ),
            Tool(
                name="statistics",
                func=self.calculate_statistics,
                description="计算数值列的统计信息，如均值，标准差，分位数等"
            ),
            Tool(
                name = "data_filter",
                func=self.filter_data,
                description="根据地区，产品类别等条件过滤数据"
            ),
            Tool(
                name="visualization",
                func=self.create_visualization,
                description="创建各种可视化图表"
            )
        ]