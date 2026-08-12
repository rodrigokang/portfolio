import math

class RegularPolygon:
    def __init__(self, base, n):
        self.base = base
        self.n = n

    def __str__(self):
        return "I am a polygon of {} sides of length {}".format(self.n, self.base)

    @property
    def apothem(self):
        return self.base / (2 * math.tan(math.pi / self.n))

    @property
    def area(self):
        return 0.5 * self.perimeter * self.apothem

    @property
    def perimeter(self):
        return self.base * self.n


class Triangle(RegularPolygon):
    def __init__(self, base):
        super().__init__(base, 3)


class Square(RegularPolygon):
    def __init__(self, base):
        super().__init__(base, 4)


class Pentagon(RegularPolygon):
    def __init__(self, base):
        super().__init__(base, 5)


class Tetrahedron(Triangle):
    @property
    def area(self):
        return math.sqrt(3) * self.base ** 2

    @property
    def volume(self):
        return (math.sqrt(2) / 12) * self.base ** 3


class Cube(Square):
    @property
    def volume(self):
        return self.base ** 3


class Circle:
    def __init__(self, r):
        self.radius = r

    @property
    def diameter(self):
        return 2 * self.radius

    @property
    def perimeter(self):
        return 2 * math.pi * self.radius

    @property
    def area(self):
        return math.pi * self.radius ** 2


class Cylinder(Circle):
    def __init__(self, r, h):
        super().__init__(r)
        self.height = h

    @property
    def area(self):
        circle_area = super().area
        square_area = self.perimeter * self.height
        return 2 * circle_area + square_area

    @property
    def volume(self):
        return super().area * self.height


if __name__ == "__main__":
    # Testing the classes
    print("Testing RegularPolygon")
    poly = RegularPolygon(4, 6)
    print(poly)
    print("Apothem:", poly.apothem)
    print("Area:", poly.area)
    print("Perimeter:", poly.perimeter)

    print("\nTesting Triangle")
    triangle = Triangle(5)
    print(triangle)
    print("Apothem:", triangle.apothem)
    print("Area:", triangle.area)
    print("Perimeter:", triangle.perimeter)

    print("\nTesting Square")
    square = Square(5)
    print(square)
    print("Apothem:", square.apothem)
    print("Area:", square.area)
    print("Perimeter:", square.perimeter)

    print("\nTesting Pentagon")
    pentagon = Pentagon(5)
    print(pentagon)
    print("Apothem:", pentagon.apothem)
    print("Area:", pentagon.area)
    print("Perimeter:", pentagon.perimeter)

    print("\nTesting Tetrahedron")
    tetrahedron = Tetrahedron(5)
    print(tetrahedron)
    print("Volume:", tetrahedron.volume)

    print("\nTesting Cube")
    cube = Cube(5)
    print(cube)
    print("Volume:", cube.volume)

    print("\nTesting Circle")
    circle = Circle(5)
    print("Diameter:", circle.diameter)
    print("Perimeter:", circle.perimeter)
    print("Area:", circle.area)

    print("\nTesting Cylinder")
    cylinder = Cylinder(5, 10)
    print(cylinder)
    print("Area:", cylinder.area)
    print("Volume:", cylinder.volume)