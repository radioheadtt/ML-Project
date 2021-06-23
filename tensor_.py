from .base import Tensor_

class tensor_(Tensor_):
    """
        Tracer wrapper to support bounded methods
    """
    def __init__(self, *args, **kwargs):
        super(tensor_, self).__init__(*args, **kwargs)
            
    
    def from_Tensor_(self, old_Tensor_ : Tensor_):
        try:
            assert isinstance(old_Tensor_, Tensor_)
        except AssertionError:
            raise AssertionError(f"expect \'last\' to by Tensor_ type, get {type(old_Tensor_)}")
        self._last_node = old_Tensor_.before()
        self._is_leaf   = old_Tensor_.is_leaf()
        self._required  = old_Tensor_.ask_grad()
        self.data = old_Tensor_.data
        return self
        
    def Gather(self, index):
        return tensor_().from_Tensor_(self.base_Gather(index))
        
    def View(self, *shape):
        return tensor_().from_Tensor_(self.base_View(*shape))
    
    def Sum(self):
        return tensor_().from_Tensor_(self.base_Sum())
    
    def __repr__(self):
        hints = super(Tensor_, self).__repr__()
        hints = "tensor_" + hints[6:]
        return hints