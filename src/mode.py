##
## EPITECH PROJECT, 2026
## G-AIA-401-PAR-4-1-starttrek-16
## File description:
## mode
##

class ModeMeta(type):
    def __instancecheck__(cls, instance):
        return cls.__subclasscheck__(type(instance))

    def __subclasscheck__(cls, subclass):
        return (hasattr(subclass, 'load_agent') and
                callable(subclass.load_agent) and
                hasattr(subclass, 'step') and
                callable(subclass.step) and
                hasattr(subclass, 'end') and
                callable(subclass.end))

class Mode(metaclass=ModeMeta):
    pass