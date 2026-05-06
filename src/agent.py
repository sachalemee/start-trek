##
## EPITECH PROJECT, 2026
## StartTrek
## File description:
## IAgent
##

class AgentMeta(type):
    def __instancecheck__(cls, instance):
        return cls.__subclasscheck__(type(instance))

    def __subclasscheck__(cls, subclass):
        return (hasattr(subclass, 'learn') and 
                callable(subclass.learn) and 
                hasattr(subclass, 'act') and 
                callable(subclass.act))

class Agent(metaclass=AgentMeta):
    pass