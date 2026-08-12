def sum(*numbers):
    """
    Function that sums the elements passed as parameters
    """
    result = 0
    for n in numbers:
        result += n
  
    return result
  
def prod(*numbers):
    """
    Function that multiplies the elements passed as parameters
    """
    result = 1
    for n in numbers:
        result *= n
  
    return result
    
def description():
    print("This module has 3 functions: ")
    print("\t- the one that displays the module's description")
    print("\t- the one that adds the numbers passed as parameters")
    print("\t- the one that multiplies the numbers passed as parameters")

# Added variables

sum1to10 = sum(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
prod1to10 = prod(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)