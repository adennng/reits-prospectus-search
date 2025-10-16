#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
招募说明书文件管理模块
负责文件名查询、数据库连接等功能
"""

import sys
import os
from typing import Optional
import pymysql

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from db_config import get_db_announcement_config


class FileManager:
    """招募说明书文件管理类"""
    
    def __init__(self):
        """初始化文件管理器"""
        self.db_config = get_db_announcement_config()
        print("[FileManager] 文件管理器初始化完成")
    
    def determine_prospectus_file(self, fund_code: str, is_expansion: bool) -> Optional[str]:
        """确定目标招募说明书文件名"""
        
        print(f"[FileManager] 查询{'扩募' if is_expansion else '首发'}招募说明书文件名...")
        
        try:
            # 建立数据库连接
            connection = self._get_db_connection()
            
            # 根据是否扩募构建不同的SQL查询
            if is_expansion:
                # 查询扩募招募说明书
                sql = f"""
                SELECT file_name, date 
                FROM processed_files 
                WHERE fund_code = '{fund_code}' 
                  AND elasticsearch_database_done = 'true'
                  AND doc_type_2 = '招募说明书'
                  AND file_name LIKE '%扩募%'
                  AND file_name NOT LIKE '%提示性%'
                ORDER BY date ASC
                LIMIT 1
                """
            else:
                # 查询首发招募说明书
                sql = f"""
                SELECT file_name, date 
                FROM processed_files 
                WHERE fund_code = '{fund_code}' 
                  AND elasticsearch_database_done = 'true'
                  AND doc_type_2 = '招募说明书'
                  AND file_name NOT LIKE '%扩募%'
                  AND file_name NOT LIKE '%提示性%'
                ORDER BY date ASC
                LIMIT 1
                """
            
            print(f"[FileManager] 执行SQL查询...")
            
            # 执行查询
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute(sql)
                results = cursor.fetchall()
            
            # 处理查询结果
            if results and len(results) > 0:
                file_name = results[0]['file_name']
                date = results[0]['date']
                print(f"[FileManager] 找到{'扩募' if is_expansion else '首发'}招募说明书: {file_name} (日期: {date})")
                return file_name
            else:
                print(f"[FileManager] 未找到基金 {fund_code} 的{'扩募' if is_expansion else '首发'}招募说明书")
                return None
                
        except Exception as e:
            print(f"[FileManager] 查询招募说明书文件名异常: {e}")
            return None
            
        finally:
            if 'connection' in locals() and connection:
                connection.close()
                print(f"[FileManager] 数据库连接已关闭")
    
    def _get_db_connection(self):
        """获取数据库连接"""
        try:
            config = self.db_config
            connection = pymysql.connect(
                host=config["host"],
                port=config["port"],
                user=config["user"],
                password=config["password"],
                database=config["database"],
                charset=config["charset"]
            )
            print(f"[FileManager] 数据库连接成功")
            return connection
        except Exception as e:
            print(f"[FileManager] 数据库连接失败: {e}")
            raise e