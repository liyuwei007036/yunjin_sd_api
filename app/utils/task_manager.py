"""
任务管理器：管理任务状态（使用SQLite3持久化）
"""
import sqlite3
import json
from typing import Dict, Optional, Union, List
from datetime import datetime
from pathlib import Path
from app.models.task import Task, TaskStatus
from app.config import Config


class TaskManager:
    """任务管理器类（使用SQLite3数据库存储）"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        初始化任务管理器
        
        Args:
            db_path: 数据库文件路径（如果为None，则使用Config.TASK_DB_PATH）
        """
        self.db_path = db_path or Config.TASK_DB_PATH
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 使用Row工厂，可以通过列名访问
        return conn
    
    def _init_database(self):
        """初始化数据库表"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 创建任务表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                result_url TEXT,
                result_urls TEXT,
                error_message TEXT,
                callback_url TEXT,
                prompt TEXT,
                negative_prompt TEXT
            )
        """)
        
        # 添加新列（如果不存在）- SQLite不支持IF NOT EXISTS，使用try-except
        try:
            cursor.execute("ALTER TABLE tasks ADD COLUMN prompt TEXT")
        except sqlite3.OperationalError:
            pass  # 列已存在
        
        try:
            cursor.execute("ALTER TABLE tasks ADD COLUMN negative_prompt TEXT")
        except sqlite3.OperationalError:
            pass  # 列已存在
        
        # 创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON tasks(status)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON tasks(created_at)
        """)
        
        conn.commit()
        conn.close()
    
    def _row_to_task(self, row: sqlite3.Row) -> Task:
        """将数据库行转换为Task对象"""
        result_urls = None
        try:
            if row['result_urls']:
                try:
                    result_urls = json.loads(row['result_urls'])
                except (json.JSONDecodeError, TypeError):
                    pass
        except (KeyError, TypeError):
            pass
        
        task = Task(
            task_id=row['task_id'],
            status=TaskStatus(row['status']),
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            result_url=row['result_url'] if 'result_url' in row.keys() else None,
            result_urls=result_urls,
            error_message=row['error_message'] if 'error_message' in row.keys() else None,
            callback_url=row['callback_url'] if 'callback_url' in row.keys() else None,
            prompt=row['prompt'] if 'prompt' in row.keys() else None,
            negative_prompt=row['negative_prompt'] if 'negative_prompt' in row.keys() else None
        )
        return task
    
    def create_task(self, task_id: str, callback_url: Optional[str] = None, 
                    prompt: Optional[str] = None, negative_prompt: Optional[str] = None) -> Task:
        """
        创建新任务
        
        Args:
            task_id: 任务ID
            callback_url: 回调URL
            prompt: 提示词
            negative_prompt: 负面提示词
        
        Returns:
            创建的任务对象
        """
        now = datetime.now().isoformat()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO tasks (task_id, status, created_at, updated_at, callback_url, prompt, negative_prompt)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (task_id, TaskStatus.PENDING.value, now, now, callback_url, prompt, negative_prompt))
        
        conn.commit()
        conn.close()
        
        # 返回创建的任务对象
        task = Task(
            task_id=task_id,
            status=TaskStatus.PENDING,
            created_at=datetime.fromisoformat(now),
            updated_at=datetime.fromisoformat(now),
            callback_url=callback_url,
            prompt=prompt,
            negative_prompt=negative_prompt
        )
        return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        获取任务
        
        Args:
            task_id: 任务ID
        
        Returns:
            任务对象，如果不存在返回None
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM tasks WHERE task_id = ?
        """, (task_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_task(row)
        return None
    
    def update_status(self, task_id: str, status: TaskStatus):
        """
        更新任务状态
        
        Args:
            task_id: 任务ID
            status: 任务状态
        """
        updated_at = datetime.now().isoformat()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE tasks 
            SET status = ?, updated_at = ?
            WHERE task_id = ?
        """, (status.value, updated_at, task_id))
        
        conn.commit()
        conn.close()
    
    def complete_task(self, task_id: str, url: Union[str, List[str]]):
        """
        完成任务
        
        Args:
            task_id: 任务ID
            url: 图片URL或URL列表
        """
        updated_at = datetime.now().isoformat()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if isinstance(url, list):
            # 多张图片：存储到result_urls（JSON格式）
            result_urls_json = json.dumps(url)
            cursor.execute("""
                UPDATE tasks 
                SET status = ?, result_urls = ?, updated_at = ?
                WHERE task_id = ?
            """, (TaskStatus.COMPLETED.value, result_urls_json, updated_at, task_id))
        else:
            # 单张图片：存储到result_url
            cursor.execute("""
                UPDATE tasks 
                SET status = ?, result_url = ?, updated_at = ?
                WHERE task_id = ?
            """, (TaskStatus.COMPLETED.value, url, updated_at, task_id))
        
        conn.commit()
        conn.close()
    
    def fail_task(self, task_id: str, error_message: str):
        """
        任务失败
        
        Args:
            task_id: 任务ID
            error_message: 错误信息
        """
        updated_at = datetime.now().isoformat()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE tasks 
            SET status = ?, error_message = ?, updated_at = ?
            WHERE task_id = ?
        """, (TaskStatus.FAILED.value, error_message, updated_at, task_id))
        
        conn.commit()
        conn.close()
    
    def get_all_tasks(self, status: Optional[TaskStatus] = None) -> Dict[str, Task]:
        """
        获取所有任务（可选按状态筛选）
        
        Args:
            status: 可选的状态筛选
        
        Returns:
            任务字典
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if status:
            cursor.execute("""
                SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC
            """, (status.value,))
        else:
            cursor.execute("""
                SELECT * FROM tasks ORDER BY created_at DESC
            """)
        
        rows = cursor.fetchall()
        conn.close()
        
        tasks = {}
        for row in rows:
            task = self._row_to_task(row)
            tasks[task.task_id] = task
        
        return tasks


# 全局任务管理器实例
task_manager = TaskManager()
