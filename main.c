#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <stdint.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>

// implikacja gate
const float train[][3] = {
    {0, 0, 1},
    {0, 1, 1},
    {1, 0, 0},
    {1, 1, 1}
};

const size_t row_length = sizeof(train[0]) / sizeof(train[0][0]);
const size_t array_length = sizeof(train) / sizeof(train[0]);

float rand_float(void)
{
    return (float)rand() / (float)RAND_MAX;
}

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

float predict(const float input[row_length - 1],
              const float weights[row_length - 1],
              float bias)
{
    float z = bias;
    for (size_t j = 0; j < row_length - 1; ++j)
        z += input[j] * weights[j];

    return sigmoidf(z);
}

float cost(const float data[][row_length],
           const float weights[row_length - 1],
           float bias)
{
    float result = 0.0f;
    for (size_t i = 0; i < array_length; ++i)
    {
        float prediction = predict(data[i], weights, bias);
        float error = prediction - data[i][row_length - 1];
        result += error * error;
    }
    result /= array_length;
    return result;
}

void gradient(const float data[][row_length],
              const float weights[row_length - 1],
              float bias,
              float grad[row_length - 1],
              float *bias_grad)
{
    for (size_t j = 0; j < row_length - 1; ++j)
        grad[j] = 0.0f;
    *bias_grad = 0.0f;

    // delta = (f(x+e) - f(x))/e
    for (size_t i = 0; i < array_length; ++i)
    {
        float prediction = predict(data[i], weights, bias);
        float error = prediction - data[i][row_length - 1];
        float delta = 2.0f * error * prediction * (1.0f - prediction)
                   / (float)array_length;

        for (size_t j = 0; j < row_length - 1; ++j)
            grad[j] += delta * data[i][j];
        *bias_grad += delta;
    }
}

// void check_by_input(const float weights[row_length - 1], float bias)
// {
//     float input[row_length - 1];

//     for (size_t j = 0; j < row_length - 1; ++j)
//     {
//         char line[128];
//         char *end;

//         printf("Enter input %zu: ", j + 1);
//         if (fgets(line, sizeof line, stdin) == NULL)
//         {
//             fputs("Could not read input.\n", stderr);
//             exit(0);
//         }

//         errno = 0;
//         float input_value = strtof(line, &end);
//         if (end == line)
//         {
//             fputs("Invalid input; enter a finite number.\n", stderr);
//             exit(0);
//         }

//         while (isspace((unsigned char)*end))
//             ++end;
//         if (*end != '\0' || errno == ERANGE || !isfinite(input_value))
//         {
//             fputs("Invalid input; enter a finite number.\n", stderr);
//             exit(0);
//         }

//         input[j] = input_value;
//     }

//     float probability = predict(input, weights, bias);
//     printf("Probability: %.6f\n", probability);
//     printf("Class: %d\n", probability >= 0.5f ? 1 : 0);
// }

int main(void)
{
    srand((unsigned int)time(NULL));

    float rate = 1e-2;
    float weights[row_length - 1];
    float grad[row_length - 1];
    float bias = 0.0f;
    float bias_grad;
    for (size_t j = 0; j < row_length - 1; ++j)
        weights[j] = j == 0 ? 1.0f : 0.0f;

    printf("cost= %.8f\n", cost(train, weights, bias));
    for (size_t i = 0; i < 100000; i++)
    {
        gradient(train, weights, bias, grad, &bias_grad);
        for (size_t j = 0; j < row_length - 1; ++j)
            weights[j] -= rate * grad[j];
        bias -= rate * bias_grad;

        // printf("cost= %.8f, weights=", cost(train, weights, bias));
        // for (size_t j = 0; j < row_length - 1; ++j)
        //     printf(" %.8f", weights[j]);
        // printf(", bias= %.8f", bias);
        // putchar('\n');
    }

    float input[2] = {0.0, 0.0};
    for (size_t i = 0; i < 2; i++)
    {
        input[0] = (float)i;
        for (size_t j = 0; j < 2; j++)
        {
            input[1] = (float)j;
            printf("%zu & %zu = %d\n", i, j, predict(input, weights, bias) >= 0.5f ? 1 : 0);
        }
           
    }
    

    return 0;
}
