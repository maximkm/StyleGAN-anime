import collections

class ClassRegistry(collections.UserDict):
    def add_to_registry(self, name):
        def add_class_by_name(cls):
            self[name] = cls
            return cls

        return add_class_by_name
