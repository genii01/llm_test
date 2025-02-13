class Demo1:
    def __init__(self, name):
        self.name = name

    def say_hello(self):
        print(f"Hello, {self.name}!")


class Demo2(Demo1):
    def __init__(self, name, age):
        super().__init__(name)
        self.age = age

    def say_version1(self):
        self.say_hello()
        print(f"Hello, {self.name}! I am {self.age} years old.")


demo_instance = Demo2("johnn", 20)
demo_instance.say_version1()
