import torch
from pathlib import Path
from enum import Enum

'''
this is another important process. since we use Kaggle we cant really fully train the model
so, we have to make checkpoints , so im thinking about keeping check points by 1000 , that is replece by another 1000
each 5000 epocs we take iterative check points we have like 5000,10000 ,15000 checkpoints.
and we also keep best model. im just trying this . dont know if that is useful.
'''

class TrainingPhase(Enum):
    PRETRAIN = "pretrain"
    SRGAN = "srgan"


class CheckpointHandler:
    def __init__(self,primary_path,phase=TrainingPhase.SRGAN):
        self.base_dir = Path(primary_path)
        self.phase = phase
        
        self.latest_dir = self.base_dir / phase.value / 'latest'
        self.best_dir = self.base_dir / phase.value / 'best'
        self.numbered_dir = self.base_dir / phase.value / 'numbered'
        
        for dir_path in [self.latest_dir, self.best_dir, self.numbered_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.best_psnr = 0.0

    def save_checkpoint(self, generator,g_optimizer=None,g_scheduler = None, discriminator=None, 
                        d_optimizer=None, d_scheduler = None,
                       iteration=0, psnr=None, is_best=False):
        
        if self.phase == TrainingPhase.PRETRAIN:
            checkpoint = {
                'iteration': iteration,
                'generator_state': generator.state_dict(),
                'g_optimizer_state': g_optimizer.state_dict() if g_optimizer else None,
                'g_scheduler_state': g_scheduler .state_dict() if g_scheduler else None
            }
        else:  
            checkpoint = {
                'iteration': iteration,
                'generator_state': generator.state_dict(),
                'discriminator_state': discriminator.state_dict() if discriminator else None,
                'g_optimizer_state': g_optimizer.state_dict() if g_optimizer else None,
                'd_optimizer_state': d_optimizer.state_dict() if d_optimizer else None,
                'g_scheduler_state': g_scheduler .state_dict() if g_scheduler else None,
                'd_scheduler_state': d_scheduler .state_dict() if d_scheduler else None
            }

        latest_path = self.latest_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)

        if iteration % 5000 == 0:
            numbered_path = self.numbered_dir / f'checkpoint_{iteration}.pt'
            torch.save(checkpoint, numbered_path)

        if is_best and psnr is not None and psnr > self.best_psnr:
            self.best_psnr = psnr
            best_path = self.best_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)


    def load_checkpoint(self, generator,g_optimizer=None,g_scheduler = None, discriminator=None, 
                        d_optimizer=None,d_scheduler = None, 
                       checkpoint_type='latest'):
        
        if checkpoint_type == 'latest':
            checkpoint_path = self.latest_dir / 'latest_checkpoint.pt'
        elif checkpoint_type == 'best':
            checkpoint_path = self.best_dir / 'best_model.pt'
        else:
            checkpoint_path = self.numbered_dir / f'checkpoint_{checkpoint_type}.pt'

        if not checkpoint_path.exists():
            print(f"No checkpoint found at {checkpoint_path}")
            return 0

        checkpoint = torch.load(checkpoint_path)

        generator.load_state_dict(checkpoint['generator_state'])

        if g_scheduler and 'g_scheduler_state' in checkpoint:
            g_scheduler.load_state_dict(checkpoint['g_scheduler_state'])
            
        if g_optimizer and 'g_optimizer_state' in checkpoint:
            g_optimizer.load_state_dict(checkpoint['g_optimizer_state'])
        
        if self.phase == TrainingPhase.SRGAN:
            if discriminator and 'discriminator_state' in checkpoint:
                discriminator.load_state_dict(checkpoint['discriminator_state'])
            
            if d_optimizer and 'd_optimizer_state' in checkpoint:
                d_optimizer.load_state_dict(checkpoint['d_optimizer_state'])

            if d_scheduler and 'd_scheduler_state' in checkpoint:
                d_scheduler.load_state_dict(checkpoint['d_scheduler_state'])
        
        
        return checkpoint['iteration']
    
    def clean_old_checkpoints(self, keep_last_n=3):
        
        checkpoint_files = sorted(list(self.numbered_dir.glob('checkpoint_*.pt')))
        
        for checkpoint_file in checkpoint_files[:-keep_last_n]:
            checkpoint_file.unlink()

    