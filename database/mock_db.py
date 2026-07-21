import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / 'db' / 'employees.db'

def get_connection(db_path:Path = DB_PATH) -> sqlite3.Connection:
    """ 业务运行时连接函数，仅连接并开启外健 """
    if not db_path.exists():
        raise FileNotFoundError(
            f'[错误] 数据库文件夹未找到：{db_path} \n'
            f'请先在终端手动运行一遍初始化版本：python database/mock_db.py \n'
        )

    conn = sqlite3.connect(str(db_path))
    conn.execute('PRAGMA foreign_keys = ON')
    return conn

def init_db(db_path:Path = DB_PATH) -> sqlite3.Connection:
    """ 数据库初始化语数据落盘(仅手动单次运行) """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 连接数据库
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute('PRAGMA foreign_keys = ON')
    cursor = conn.cursor()      # 游标

    # 创建员工表
    cursor.execute("""
    create table if not exists employees (
        uid text primary key,           -- 员工唯一标识
        name text not null,             -- 员工姓名
        rank text not null,             -- 职级（p3、p4）
        location text not null,         -- 工作地点（城市名称）
        seniority integer not null,     -- 入职年限
        base_salary integer not null    -- 基本工资
    )
    """)

    # 创建假期表
    cursor.execute("""
     create table if not exists leave_balance (
         uid text primary key,                          -- 员工唯一标识（外键关联 employees.id）
         annual_leave_remaining integer not null,       -- 剩余年假天数
         sick_leave_remaining integer not null,         -- 剩余病假天数
         foreign key (uid) references employees( uid)   
     )
     """)

    # 清空旧数据（确保幂等性）
    cursor.execute(""" delete from employees""")
    cursor.execute(""" delete from leave_balance where 1=1""")

    # 添加数据
    test_employees = [
        ('1001', '张三', 'P5', '北京', 2, 10000),
        ('1002', '李四', 'P4', '北京', 4, 9000),
        ('1003', '王五', 'P7', '北京', 5, 35000),
        ('1004', '赵六', 'P3', '北京', 0, 7500),
    ]
    test_balance = [
        ('1001', 6, 10),
        ('1002', 7, 12),
        ('1003', 14, 15),
        ('1004', 2, 5),
    ]

    # executemany，多条元组列表
    cursor.executemany(""" insert into employees values (?,?,?,?,?,?)""", test_employees)
    cursor.executemany(""" insert into leave_balance values (?,?,?)""", test_balance)

    conn.commit()

    print('[成功] 实体数据库已成功落盘')
    print(f'数据库路径：{db_path}')
    return conn

def query_db(conn:sqlite3.Connection, sql:str, params:tuple=()):
    """ 通用查询函数 """
    cursor = conn.cursor()
    cursor.execute(sql, params)
    columns = [col[0] for col in cursor.description]    # 获取元数据, col[0]包含表的列名
    # cursor.fetchall()：获取所有行;  zip(columns, row)：将列名和值配对；  dict：转为字典
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def close_db(conn:sqlite3.Connection):
    """ 安全关闭数据库 """
    if conn:
        conn.close()
        print('数据库连接已安全关闭')

if __name__ == '__main__':
    print('正在执行数据库手动初始化操作')
    standalone_conn = init_db()
    close_db(standalone_conn)
