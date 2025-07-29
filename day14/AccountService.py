import FileDao
import DbDao

class AccountService:
    def __init__(self,data=None):
        if not data: self.repo = FileDao.FileService()
        elif data == "db": self.repo = DbDao.MySQLService()
    def create_account(self,account_id, name, balance):
        self.repo.openAccount(account_id,name,balance)
    def read_accounts(self):
        self.repo.viewAccounts()
    def update_account(self,account_id,**kwargs):
        self.repo.updateAccount(account_id,**kwargs)
    def delete_account(self,account_id):
        self.repo.suspendAccount(account_id)
    # def __enter__(self):
    #     # Automatically called when entering 'with' block
    #     return self
    # def __exit__(self):
    #     if hasattr(self.repo,"__exit__"):
    #         self.repo.__exit__(self, exc_type, exc_val, exc_tb)