import sys
import pymysql
from colorama import Fore, Back, Style


class MySQLConnection(object):

    def __init__(self, host, user, password, database):
        self.__cursor = None
        self.__connection_obj = None
        self.host = host
        self.user = user
        self.password = password
        self.database = database

        banner = """
        ======================================================================
            MySQL Connection - Version 1.0 - Start of connection
        =======================================================================
        Author: Brown Sugar
        Description: This is a sample script
        """
        print(Fore.YELLOW + '='*80)
        print(Fore.YELLOW + 'MySQL Connection - Version 1.0 - Start of connection')
        print('='*80)
        # print('Author: Brown Sugar')
        # print('Description: This is a sample script')

        try:
            connection_obj = pymysql.connect(host=host, port=3306, user=user, password=password,database=database)
        except ConnectionError as e:
            print(f'数据库连接错误：{e}')
            sys.exit()
        self.__connection_obj = connection_obj
        # print('数据库连接连接成功')

    def execute(self, sql):
        if sql is None or sql == '':
            print('sql脚本不能为空')
            sys.exit()

        # 创建游标对象
        cursor = self.__connection_obj.cursor()
        try:
            cursor.execute(sql)
        except Exception as e:
            print(f'数据库SQL执行错误：{e}')

        self.__cursor = cursor

        # 获取全部结果集
        return cursor.fetchall()

    # def __str__(self):
    def __del__(self):
        # 关闭游标和连接
        # print(self)
        self.__cursor.close()
        self.__connection_obj.close()

        print('='*80)
        print('MySQL Connection - Version 1.0 - Connection destruction')
        print('='*80)


def main():
    # 创建连接对象
    conn = MySQLConnection('124.70.78.122', 'root', '!@#123qwe', 'bs')
    result = conn.execute('SELECT * FROM amm_user')
    print(result)
    del conn


if __name__ == '__main__':
    main()
