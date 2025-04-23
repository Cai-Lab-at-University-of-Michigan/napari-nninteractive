import os
import stat
from filelock import FileLock

class GPUMemoryLock:
    def __init__(self, memory_per_lock=12):  # 12GB per lock
        self.memory_per_lock = memory_per_lock  # GB
        
        # Get the actual physical GPU IDs, not the CUDA_VISIBLE_DEVICES mapping
        self.physical_gpu_ids = self._get_physical_gpu_ids()
        
        # Create locks directory with read/write permissions for everyone
        locks_dir = "/tmp/gpu_locks"
        os.makedirs(locks_dir, exist_ok=True)
        
        # Set directory permissions to 777 (rwxrwxrwx)
        try:
            os.chmod(locks_dir, 0o777)
        except PermissionError:
            print(f"Warning: Could not set permissions on {locks_dir}. You may need admin rights.")
        
        # Get actual memory for each GPU
        self.gpu_memory = self._get_gpu_memory()
        
        # Create locks based on actual memory, using physical GPU IDs
        self.lock_files = {}
        for idx, physical_id in enumerate(self.physical_gpu_ids):
            gpu_mem = self.gpu_memory.get(physical_id, 12)  # Default to 12GB
            num_locks = max(1, int(gpu_mem / memory_per_lock))
            
            for lock_id in range(num_locks):
                # Use physical GPU ID for lock file names
                lock_path = f"{locks_dir}/physical_gpu_{physical_id}_mem_{lock_id}.lock"
                self.lock_files[(idx, lock_id)] = lock_path
                
                # Create empty lock file with read/write permissions if it doesn't exist
                self._ensure_lock_file_has_rw_permissions(lock_path)
    
    def _ensure_lock_file_has_rw_permissions(self, lock_path):
        """Ensure the lock file exists with read/write permissions (666)"""
        try:
            # Create the file if it doesn't exist
            if not os.path.exists(lock_path):
                with open(lock_path, 'w') as f:
                    pass
                
            # Set permissions to 666 (rw-rw-rw-) - read/write for all users
            os.chmod(lock_path, 0o666)
            
        except PermissionError:
            print(f"Warning: Could not set permissions on {lock_path}")
    
    def _get_physical_gpu_ids(self):
        """
        Get the actual physical GPU IDs from CUDA_VISIBLE_DEVICES
        If CUDA_VISIBLE_DEVICES="2,3", we want [2,3] not [0,1]
        """
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_devices:
            # Parse the physical GPU IDs
            return [int(x.strip()) for x in cuda_devices.split(",") if x.strip()]
        
        # If CUDA_VISIBLE_DEVICES isn't set, get all available GPUs
        try:
            import torch
            return list(range(torch.cuda.device_count()))
        except:
            try:
                import subprocess
                output = subprocess.check_output(['nvidia-smi', '-L'], universal_newlines=True)
                # Count the number of GPUs
                count = output.count('GPU ')
                return list(range(count))
            except:
                return [0]  # Default to GPU 0
    
    def _get_gpu_memory(self):
        """Read actual GPU memory for each device"""
        gpu_memory = {}
        try:
            import subprocess
            output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total', '--format=csv,nounits,noheader'])
            for line in output.decode('utf-8').strip().split('\n'):
                if line.strip():
                    idx, mem = line.split(',')
                    physical_id = int(idx)
                    # Convert MB to GB
                    gpu_memory[physical_id] = float(mem) / 1024
        except:
            # Fallback - assign default values
            for physical_id in self.physical_gpu_ids:
                gpu_memory[physical_id] = 12  # Default to 12GB
        
        return gpu_memory
    
    def acquire_memory(self, memory_needed):
        """Acquire locks for the specified amount of memory"""
        locks_needed = max(1, int((memory_needed + self.memory_per_lock - 1) / self.memory_per_lock))
        
        # Try each assigned GPU
        for idx, physical_id in enumerate(self.physical_gpu_ids):
            num_locks = sum(1 for k in self.lock_files.keys() if k[0] == idx)
            
            if num_locks < locks_needed:
                continue
                
            for start_lock in range(num_locks - locks_needed + 1):
                acquired = []
                success = True
                
                for i in range(start_lock, start_lock + locks_needed):
                    lock_file = self.lock_files.get((idx, i))
                    
                    # Ensure the lock file has proper permissions before trying to acquire
                    self._ensure_lock_file_has_rw_permissions(lock_file)
                    
                    try:
                        lock = FileLock(lock_file, timeout=0.1)
                        lock.acquire()
                        acquired.append(lock)
                    except:
                        success = False
                        for lock in acquired:
                            lock.release()
                        acquired = []
                        break
                
                if success:
                    # Return logical index (0 in CUDA_VISIBLE_DEVICES)
                    return idx, acquired
        
        return None, []
    
    def release_locks(self, locks):
        """Release all provided locks"""
        for lock in locks:
            try:
                lock.release()
            except:
                pass
    
    def release_and_delete_locks(self, locks):
        """Release all provided locks and delete the associated lock files"""
        for lock in locks:
            try:
                # Get the lock file path before releasing
                lock_path = lock.lock_file
                
                # Release the lock
                lock.release()
                
                # Delete the lock file if it exists
                if os.path.exists(lock_path):
                    try:
                        os.remove(lock_path)
                        print(f"Deleted lock file: {lock_path}")
                    except PermissionError:
                        print(f"Warning: Could not delete lock file {lock_path}. Permission denied.")
                    except Exception as e:
                        print(f"Warning: Error deleting lock file {lock_path}: {e}")
            except Exception as e:
                print(f"Warning: Error releasing lock: {e}")

# Example usage:
if __name__ == "__main__":
    # Create a GPU memory lock manager
    gpu_lock = GPUMemoryLock(memory_per_lock=4)  # 4GB per lock
    
    # Try to acquire 8GB of GPU memory
    gpu_id, locks = gpu_lock.acquire_memory(8)
    
    if gpu_id is not None:
        print(f"Successfully acquired memory on GPU {gpu_id}")
        
        # Simulate doing work with the GPU
        import time
        print("Processing on GPU...")
        time.sleep(5)
        
        # Release locks and delete the lock files when done
        print("Releasing locks and deleting lock files...")
        gpu_lock.release_and_delete_locks(locks)
        print("Done!")
    else:
        print("Failed to acquire required GPU memory")