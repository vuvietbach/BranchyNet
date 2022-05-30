

class Registry:
    def __init__(self, name):
        self.name = name
        self._module_dict = dict()
    
    def _register_module(self, name, module):
        if name in self._module_dict.keys():
            raise KeyError(f'{name} if already in {self.name}')
        
        self._module_dict[name] = module
    
    def build(self, name, **kwargs):
        return self._module_dict.get(name)(**kwargs)
   
    def register_module(self, name, module=None):
        if module is not None:
            self._register_module(name, module)
            return module
        
        def _register(module):
            self._register_module(name, module)
            return module
        
        return _register
