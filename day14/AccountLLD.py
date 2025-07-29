from AccountService import *

if __name__ == "__main__":
    service = AccountService() # File as Data Logic
    # service = AccountService("db") # MySQL as Data Logic
    # service.create_account(1122,"Razak Mohamed S",1200)
    service.read_accounts()
    # service.update_account(1122,**{"accountHolder":"Mohamed"})
    # service.update_account(1122,**{"accountBalance":900})
    # service.update_account(1122,**{"accountHolder":"Mohamed","accountBalance":1900})
    # service.delete_account(1122)
    service.read_accounts()