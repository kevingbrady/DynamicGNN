import threading
from typing import *
from sqlite3 import Connection, connect, Error
from src.error_handling import try_except


class DatabaseAPI:
    def __init__(self, db_name):
        self.db_name = db_name
        # self._local_connections = threading.local()

    @try_except
    def _get_connection(self) -> Connection | None:

        # if not hasattr(self._local_connections, 'conn'):
        conn = connect(self.db_name, timeout=5, check_same_thread=False)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        # self._local_connections.conn = conn
        return conn

    @try_except
    def execute_read(self, query: str, params: Union[Tuple, List] = ()) -> list[Tuple]:
        conn = self._get_connection()
        cursor = conn.execute(query, params)
        result = cursor.fetchall()
        conn.close()
        return result

    @try_except
    def execute_query(self, query, params=()):

        self.cursor.execute(query, params)
        self.connection.commit()
        return self.cursor.fetchall()

    @try_except
    def execute_multi_query(self, query, data_list, params=()):

        self.cursor.executemany(query, data_list)
        self.connection.commit()
        return self.cursor.fetchall()

    @try_except
    def create_table(self, table_name, columns):
        column_definitions = ", ".join([f"{name} {data_type}" for name, data_type in columns.items()])
        query = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({column_definitions});'
        self.execute_query(query)

    @try_except
    def insert_data(self, table_name, table_columns, data):

        col = tuple(table_columns.keys())
        mask = '?,' * (len(table_columns) - 1) + '?'

        query = f'INSERT INTO "{table_name}" {col} VALUES ({mask})'
        self.execute_multi_query(query, data)

    @try_except
    def select_data(self, table_name, columns="*", condition=None):
        query = f"SELECT {columns} FROM {table_name}"
        if condition:
            query += f" WHERE {condition}"
        return self.execute_query(query)

    @try_except
    def update_data(self, table_name, data, condition):
        set_values = ", ".join([f"{key} = ?" for key in data.keys()])
        query = f"UPDATE {table_name} SET {set_values} WHERE {condition}"
        self.execute_query(query, tuple(data.values()))

    @try_except
    def delete_data(self, table_name, condition):
        query = f"DELETE FROM {table_name} WHERE {condition}"
        self.execute_query(query)

    @try_except
    def table_exists(self, table_name):
        db_table_exists = self.execute_query(
            f'SELECT * FROM sqlite_master WHERE type="table" and name="{table_name}";')
        if len(db_table_exists) > 0:
            return True
        return False
