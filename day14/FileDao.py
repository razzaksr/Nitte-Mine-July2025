import os
import pickle
from models import Account
from dotenv import load_dotenv

load_dotenv()

class FileService:
    def __init__(self):
        self.ACCOUNT_FILE = os.getenv("ACCOUNT_FILE")
    def get_msg(self,key, **kwargs): return os.getenv(key, "").format(**kwargs)
    def load_accounts(self):
        collected = []
        if os.path.exists(self.ACCOUNT_FILE):
            with open(self.ACCOUNT_FILE, 'rb') as f:
                data = pickle.load(f)
                collected.extend(each for each in data)
        return collected
    def save_accounts(self,accounts):
        with open(self.ACCOUNT_FILE, 'wb') as f: pickle.dump(accounts,f)
    def openAccount(self,no,name,balance):
        accounts = self.load_accounts()
        if any(acc.accountNo == no for acc in accounts):
            print(self.get_msg("MSG_ACCOUNT_EXISTS", accountNo=no))
            return
        accounts.append(Account(no, name, balance))
        self.save_accounts(accounts)
        print(self.get_msg("MSG_ACCOUNT_CREATED"))
    def viewAccounts(self):
        accounts = self.load_accounts()
        if not accounts:
            print(self.get_msg("MSG_NO_ACCOUNTS"))
            return
        for acc in accounts:
            print(self.get_msg("MSG_ACCOUNT_DISPLAY", accountNo=acc.accountNo, accountHolder=acc.accountHolder, accountBalance=acc.accountBalance))
    def updateAccount(self,account_id, **kwargs):
        update_fields = {k: v for k, v in kwargs.items() if v is not None}
        if not update_fields:
            print(self.get_msg("MSG_ACCOUNT_NOT_FOUND"))
            return
        accounts = self.load_accounts()
        for acc in accounts:
            if acc.accountNo == account_id:
                if "accountHolder" in update_fields:
                    acc.accountHolder = update_fields.get("accountHolder")
                if "accountBalance" in update_fields:
                    acc.accountBalance = update_fields.get("accountBalance")
                self.save_accounts(accounts)
                print(self.get_msg("MSG_ACCOUNT_UPDATED"))
                return
        print(self.get_msg("MSG_ACCOUNT_NOT_FOUND"))
    def suspendAccount(self,account_id):
        accounts = self.load_accounts()
        new_accounts = [acc for acc in accounts if acc.accountNo != account_id]
        if len(new_accounts) == len(accounts):
            print(self.get_msg("MSG_ACCOUNT_NOT_FOUND"))
        else:
            self.save_accounts(new_accounts)
            print(self.get_msg("MSG_ACCOUNT_DELETED"))