import json
import shutil
import time
import uuid
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import asdict, dataclass
from core.reasoning import Decision

@dataclass
class Transaction:
    id: str
    timestamp: float
    action: str
    target_path: str
    destination_path: Optional[str]
    backup_path: Optional[str] = None
    status: str = "pending" # pending, success, failed, rolled_back

class TransactionLog:
    """
    Logs file operations to allow rollback.
    """
    def __init__(self, log_path: str = "transaction_log.json"):
        self.log_path = Path(log_path)
        self.transactions: List[Transaction] = []
        self._load_log()
        
    def _load_log(self):
        if self.log_path.exists():
            try:
                with open(self.log_path, 'r') as f:
                    data = json.load(f)
                    for t_data in data:
                        self.transactions.append(Transaction(**t_data))
            except Exception as e:
                print(f"Error loading transaction log: {e}")
                
    def _save_log(self):
        with open(self.log_path, 'w') as f:
            json.dump([asdict(t) for t in self.transactions], f, indent=2)
            
    def add_transaction(self, transaction: Transaction):
        self.transactions.append(transaction)
        self._save_log()
        
    def update_transaction(self, transaction_id: str, status: str, backup_path: str = None):
        for t in self.transactions:
            if t.id == transaction_id:
                t.status = status
                if backup_path:
                    t.backup_path = backup_path
                break
        self._save_log()
        
    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        for t in self.transactions:
            if t.id == transaction_id:
                return t
        return None

from core.safety import SafetyGuard

class FileOperationExecutor:
    """
    Executes decisions safely with rollback capability.
    """
    
    def __init__(self, log_path: str = "transaction_log.json", backup_dir: str = ".ai_os_backups", workspace_root: str = "."):
        self.log = TransactionLog(log_path)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.safety = SafetyGuard(workspace_root)
        
    def execute(self, decision: Decision) -> bool:
        """
        Executes a single decision.
        """
        # Resolve paths to absolute to avoid ambiguity
        target = Path(decision.target_path).resolve()
        dest = None
        if decision.destination_path:
            dest = Path(decision.destination_path).resolve()

        # Safety Check
        # We pass the original decision, but safety guard resolves internally too.
        is_safe, reason = self.safety.validate_decision(decision)
        if not is_safe:
            print(f"Safety Check Failed for {decision.action} on {target}: {reason}")
            return False

        tx_id = str(uuid.uuid4())
        tx = Transaction(
            id=tx_id,
            timestamp=time.time(),
            action=decision.action,
            target_path=str(target),
            destination_path=str(dest) if dest else None
        )
        self.log.add_transaction(tx)
        
        try:
            if not target.exists():
                raise FileNotFoundError(f"Target file {target} not found.")
                
            if decision.action == "move" or decision.action == "rename":
                if not dest:
                    raise ValueError("Destination path required for move/rename")
                    
                dest.parent.mkdir(parents=True, exist_ok=True)
                
                # Check if dest exists
                if dest.exists():
                    raise FileExistsError(f"Destination {dest} already exists.")
                
                shutil.move(target, dest)
                self.log.update_transaction(tx_id, "success")
                return True
                
            elif decision.action == "delete":
                # Safe delete: move to backup
                backup_path = self.backup_dir / f"{target.name}_{tx_id}"
                shutil.move(target, backup_path)
                self.log.update_transaction(tx_id, "success", str(backup_path))
                return True
                
            elif decision.action == "group_files":
                 # Treat as move
                if not dest:
                    raise ValueError("Destination path required for group_files")

                # If dest is a directory (which it should be for group_files), append filename
                # But decision.destination_path usually points to the target directory?
                # The original code did: dest = Path(decision.destination_path) / target.name
                # Let's check if the decision.destination_path is the directory or the full file path.
                # Usually group_files implies moving into a directory.
                
                final_dest = dest / target.name
                final_dest.parent.mkdir(parents=True, exist_ok=True)
                
                if final_dest.exists():
                     raise FileExistsError(f"Destination {final_dest} already exists.")
                     
                shutil.move(target, final_dest)
                # Update transaction with actual destination path
                tx.destination_path = str(final_dest)
                self.log.update_transaction(tx_id, "success")
                return True
                
            else:
                print(f"Unknown action: {decision.action}")
                self.log.update_transaction(tx_id, "failed")
                return False
                
        except Exception as e:
            print(f"Error executing {decision.action} on {target}: {e}")
            self.log.update_transaction(tx_id, "failed")
            return False
            
    def rollback(self, transaction_id: str) -> bool:
        """
        Rolls back a specific transaction.
        """
        tx = self.log.get_transaction(transaction_id)
        if not tx:
            print(f"Transaction {transaction_id} not found.")
            return False
            
        if tx.status != "success":
            print(f"Cannot rollback transaction {transaction_id} with status {tx.status}.")
            return False
            
        try:
            if tx.action == "move" or tx.action == "rename" or tx.action == "group_files":
                # Move back from destination to target
                src = Path(tx.destination_path)
                dest = Path(tx.target_path)
                
                if not src.exists():
                    print(f"Cannot rollback: Current file {src} not found.")
                    return False
                    
                if dest.exists():
                    print(f"Cannot rollback: Original location {dest} is occupied.")
                    return False
                    
                shutil.move(str(src), str(dest))
                self.log.update_transaction(tx.id, "rolled_back")
                return True
                
            elif tx.action == "delete":
                # Restore from backup
                if not tx.backup_path:
                    print("No backup path found for delete transaction.")
                    return False
                    
                backup = Path(tx.backup_path)
                dest = Path(tx.target_path)
                
                if not backup.exists():
                    print(f"Backup file {backup} not found.")
                    return False
                    
                shutil.move(str(backup), str(dest))
                self.log.update_transaction(tx.id, "rolled_back")
                return True
                
            return False
            
        except Exception as e:
            print(f"Error rolling back {transaction_id}: {e}")
            return False
