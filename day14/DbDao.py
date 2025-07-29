from mysql.connector import *
from dotenv import load_dotenv
import os
from models import Account

load_dotenv()

class MySQLService:
    def __init__(self):
        self.conn = connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DB")
        )
        self.cursor = self.conn.cursor(dictionary=True)
        self.table_name = os.getenv("MYSQL_TABLE")
        self.checkTable()
    def get_msg(self,key, **kwargs): return os.getenv(key, "").format(**kwargs)
    def checkTable(self):
        create_table_query = """
        CREATE TABLE IF NOT EXISTS accounts (
            accountNo BIGINT PRIMARY KEY,
            accountHolder VARCHAR(255),
            accountBalance FLOAT
        )
        """
        self.cursor.execute(create_table_query)
        self.conn.commit()
        # print("Table has created")
    def __enter__(self):
        # Automatically called when entering 'with' block
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Automatically called when exiting 'with' block
        self.cursor.close()
        self.conn.close()
    def openAccount(self,no,name,balance):
        try:
            self.cursor.execute(
            f"INSERT INTO {self.table_name} VALUES (%s, %s, %s)",
            (no, name, balance)
            )
            self.conn.commit()
            print(self.get_msg("MSG_ACCOUNT_CREATED"))
        except Exception as e:
            # if e.errno == 1062:
            #     print("Duplicate entry detected. Skipping insert.")
            print(self.get_msg("MSG_ACCOUNT_EXISTS", accountNo=no))
    def viewAccounts(self):
        self.cursor.execute("select * from "+self.table_name)
        accounts = self.cursor.fetchall()
        for acc in accounts:
            print(self.get_msg("MSG_ACCOUNT_DISPLAY", accountNo=acc["accountNo"], accountHolder=acc["accountHolder"], accountBalance=acc["accountBalance"]))
    def updateAccount(self,account_id, **kwargs):
        # updates = ", ".join(f"{k}={v}" if {k}!="accountHolder" else f"{k}={v}" for k,v in kwargs.items())
        updates = ", ".join(
            f"{k}='{v}'" if k == "accountHolder" else f"{k}={v}"
            for k, v in kwargs.items()
        )
        try:
            self.cursor.execute(
                f"UPDATE {self.table_name} SET {updates} WHERE accountNo="+str(account_id)
            )
            self.conn.commit()
            print(self.get_msg("MSG_ACCOUNT_UPDATED"))
        except Exception as e:
            print(e)
            print(self.get_msg("MSG_ACCOUNT_NOT_FOUND"))
    def suspendAccount(self,account_id):
        try:
            self.cursor.execute(f"delete from {self.table_name} where accountNo={account_id}")
            if self.cursor.rowcount == 0: print(self.get_msg("MSG_ACCOUNT_NOT_FOUND"))
            else:
                self.conn.commit()
                print(self.get_msg("MSG_ACCOUNT_DELETED"))
        except Exception as e:
            print(self.get_msg("MSG_ACCOUNT_NOT_FOUND"))
    # since service layers self.repo has been instantiate the dbdao only once it wouldn't called like below execution commneted
    # this manual call necessary
    def close(self):
        self.cursor.close()
        self.conn.close()
        
# mysql = MySQLService()
# Conext Manager: this enable __enter and __exit calls automatically so that close connection would be executed well
# with MySQLService() as mysql:
#     # mysql.openAccount(689,"Razak Mohamed S",12888.3)
#     mysql.viewAccounts()
#     # mysql.updateAccount(689,**{"accountBalance":10000})
#     # mysql.updateAccount(689,**{"accountHolder":"Mohamed Razak"})
#     # mysql.updateAccount(689,**{"accountHolder":"Razak Mohamed S","accountBalance":2000})
#     # mysql.suspendAccount(689)
#     mysql.viewAccounts()