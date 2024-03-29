#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
// Print a pattern of numbers from 1 to n as shown below. Each of the numbers is separated by a single space.

//                             4 4 4 4 4 4 4
//                             4 3 3 3 3 3 4
//                             4 3 2 2 2 3 4
//                             4 3 2 1 2 3 4
//                             4 3 2 2 2 3 4
//                             4 3 3 3 3 3 4
//                             4 4 4 4 4 4 4

int *create_array(int numero_de_elementos, int n)
{
    int *array = malloc(numero_de_elementos * sizeof(int));

    for (int i = 0; i < numero_de_elementos; i++)
    {
        array[i] = n;
    }
    return array;
}

void print_line(int *array, int line_size, int subtract, int which, int stop, int n)
{
    for (int i = 0; i < line_size; i++)
    {
        if (i >= which && i < stop)
        {
            array[i] = n - subtract;
        }
        printf("%d ", array[i]);
    }
    printf("\n");
}

int main()
{
    int n = 9;
    const int num_colunas = (n * 2) - 1;

    // Complete the code to print the pattern.
    
    int subtract = 0; // How much do i need to subtract
    int which = 0;    // At which element do i need to start subtracting

    int *line = create_array(num_colunas, n);
    int stop = num_colunas; // At wich element do i need to stop subtracting

    for (int iline = 0; iline < num_colunas; iline++)
    {
        if (iline < ceil(num_colunas/2))
        {
            print_line(line, num_colunas, subtract, which, stop, n);
            
            subtract++;
            which++;
            stop--;
        }else{
            print_line(line, num_colunas, subtract, which, stop, n);
            
            subtract--;
            which--;
            stop++; 
        }
    }
    free(line);
    return 0;
}