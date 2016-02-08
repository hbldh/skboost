/**
 **************************************************************************************
 * @file    $classifiers.c
 * @author  Henrik Blidh
 * @version 1.0
 * @date    2012-11-23
 * @brief   Learning algorithms implemented as Numpy/C extensions for speedup.
 **************************************************************************************
 */

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL nptest_ARRAY_API
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

static char train_decision_stump_docs[] =
      "Algorithm for training an Decision Stump.\n\n"
      "Definition::\n\n"
      "  fit_decision_stump(data, weights, arg_sorted_data)\n\n"
      "Parameters::\n\n"
      "  data\n"
      "    Values for the features used in the training.\n\n"
      "  weights\n"
      "    Weights for the different data samples.\n\n"
      "  arg_sorted_data\n"
      "    The argument used to sort the data.\n\n"
      "Return::\n\n"
      "  min_value\n"
      "    The minimum value result.\n\n"
      "  best_dim\n"
      "    The dimension to use.\n\n"
      "  min_direction\n"
      "    The direction to use.\n\n"
      "  thr\n"
      "    The threshold to use.\n\n";

static PyObject *train_decision_stump(PyObject *self, PyObject *args)
{
    PyObject *data=NULL, *weights=NULL, *arg_sorted_data=NULL;
    PyObject *data_arr=NULL, *weights_arr=NULL, *arg_sorted_arr=NULL;

    /* Local variables for handling the dimensions of the ndarrays. */
    long *data_arr_dims=NULL, *weights_arr_dims=NULL, *arg_sorted_arr_dims=NULL;
    int data_arr_nd=0, weights_arr_nd=0, arg_sorted_arr_nd=0;

    /* Pointers into the nparrays. */
    npy_double *data_ptr=NULL;
    int *arg_sorted_ptr=NULL;
    npy_double *weights_ptr=NULL;

    /* Local variables used during training.*/
    double current_weight, s_pos, s_neg, min_value_tmp;
    double min_err, max_err;
    double min_value = 100.0;
    double thr=0.0;
    int max_err_ind=0, min_err_ind=0,
        min_direction=0, min_direction_tmp=0,
        min_point=0, best_dim=0;

    /* Loop counters. */
    int i, j;

    /* Parse the input to correct pointers. */
    if (!PyArg_ParseTuple(args, "OOO", &data, &weights, &arg_sorted_data))
        return NULL;
    /* data_tuple is a tuple but now with a unknown number of arrays. */

    data_arr = PyArray_FROM_OTF(data, NPY_FLOAT64, NPY_IN_ARRAY);
    if (data_arr == NULL) goto fail;

    weights_arr = PyArray_FROM_OTF(weights, NPY_FLOAT64, NPY_IN_ARRAY);
    if (weights_arr == NULL) goto fail;

    /* This will be an integer array but we want it 32/64 bit independent. */
    arg_sorted_arr = PyArray_FROM_O(arg_sorted_data);
    if (data_arr == NULL) goto fail;

    /* Dimensions of the arrays. */
    data_arr_nd         = PyArray_NDIM(data_arr);
    data_arr_dims       = PyArray_DIMS(data_arr);
    weights_arr_nd      = PyArray_NDIM(weights_arr);
    weights_arr_dims    = PyArray_DIMS(weights_arr);
    arg_sorted_arr_nd   = PyArray_NDIM(arg_sorted_arr);
    arg_sorted_arr_dims = PyArray_DIMS(arg_sorted_arr);

    if (data_arr_nd != 2)
    {
        printf("Dimension of data array shall be 2. (%d != 2).\n",
               data_arr_nd);
        goto fail;
    }

    if (weights_arr_nd != 1)
    {
        printf("Dimension of weights array shall be 1. (%d != 1).\n",
               weights_arr_nd);
        goto fail;
    }

    if (arg_sorted_arr_nd != 2)
    {
        printf("Dimension of arg sorted data shall be 2. (%d != 2).\n",
               data_arr_nd);
        goto fail;
    }

    if (data_arr_dims[1] != weights_arr_dims[0])
    {
        printf("Data and weights must have the same length. (%ld != %ld).\n",
               data_arr_dims[1], weights_arr_dims[0]);
        goto fail;
    }

    if ((data_arr_dims[0] != arg_sorted_arr_dims[0]) && (data_arr_dims[1] != arg_sorted_arr_dims[1]))
    {
        printf("Data and and argsorted must be equal in shape. ((%ld, %ld) != (%ld, %ld)).\n",
               data_arr_dims[0], data_arr_dims[1], arg_sorted_arr_dims[0], arg_sorted_arr_dims[1]);
        goto fail;
    }

    weights_ptr = (npy_double *)PyArray_GETPTR1(weights_arr, 0);
    /* Iterate over all features. */
    for(i = 0; i < data_arr_dims[0]; i++)
    {
        /* Get the pointer to the sorted data, this array is transposed compared to data. */
        s_neg = 0;
        s_pos = 0;
        /* 100 is enough since the weights sums up to 1, 2 would had been enough.*/
        max_err = -100.0;
        min_err = 100.0;

        /* Time to evaluate the result. */
        for(j = 0; j < data_arr_dims[1]; j++)
        {
            arg_sorted_ptr = (int *)PyArray_GETPTR2(arg_sorted_arr, i, j);
            current_weight = weights_ptr[arg_sorted_ptr[0]];
            if(current_weight > 0)
            {
                s_pos += current_weight;
            }
            else
            {
                s_neg += current_weight;
            }

            if((s_pos + s_neg) > max_err)
            {
                max_err = s_pos + s_neg;
                max_err_ind = j;
            }
            if((s_pos + s_neg) < min_err)
            {
                min_err = s_pos + s_neg;
                min_err_ind = j;
            }
        }

        /* Now we need to compare the positive and the negative error.*/
        /* See python code for definition on direction.*/
        if((min_err - s_neg) < (-max_err + s_pos))
        {
            min_value_tmp = min_err - s_neg;
            min_point = min_err_ind;
            min_direction_tmp = 1;
        }
        else
        {
            min_value_tmp = -max_err + s_pos;
            min_point = max_err_ind;
            min_direction_tmp = -1;
        }
        /* Check if this is the best result so far. */
        if(min_value_tmp < min_value)
        {
            min_value = min_value_tmp;
            /* We need to turn everything around if the last point is the best. */
            if(min_point == (data_arr_dims[1] -1))
            {
                min_point = 0;
                min_direction_tmp = -min_direction_tmp;
            }
            /* It is a special case if the min_point is in the beginning. */
            if(min_point == 0){
                arg_sorted_ptr = (int *)PyArray_GETPTR2(arg_sorted_arr, i, min_point);
                data_ptr = (npy_double *)PyArray_GETPTR2(data_arr, i, arg_sorted_ptr[0]);
                thr  = data_ptr[0] - data_ptr[0]*0.01;
            }
            else
            {
                /* TODO: Use get pointer less by using strides. This usage is to make it 32/64 independent. */
                arg_sorted_ptr = (int *)PyArray_GETPTR2(arg_sorted_arr, i, min_point);
                data_ptr = (npy_double *)PyArray_GETPTR2(data_arr, i, arg_sorted_ptr[0]);
                thr = data_ptr[0];
                arg_sorted_ptr = (int *)PyArray_GETPTR2(arg_sorted_arr, i, min_point+1);
                data_ptr = (npy_double *)PyArray_GETPTR2(data_arr, i, arg_sorted_ptr[0]);
                thr += data_ptr[0];
                thr /= 2;
            }
            best_dim = i;
            min_direction = min_direction_tmp;
        }
    }

    /* Decrease reference counters to avoid memory leaks. */
    Py_DECREF(data_arr);
    Py_DECREF(weights_arr);
    Py_DECREF(arg_sorted_arr);

    return Py_BuildValue("didi", min_value, best_dim, thr, min_direction);

 fail:
    /* If error occurs, decrease reference counters to avoid memory leaks. */
    Py_XDECREF(data_arr);
    Py_XDECREF(weights_arr);
    Py_XDECREF(arg_sorted_arr);
    /* Return error indication. */
    return NULL;

}


static char train_regression_stump_docs[] =
      "Algorithm for training an Regression Stump with floating point data.\n\n"
      "Definition::\n\n"
      "  fit_regression_stump(data, desired_output, weights, arg_sorted_data)\n\n"
      "Parameters::\n\n"
      "  data\n"
      "    Values for the features used in the training. (float)\n\n"
      "  desired_output\n"
      "    The desired output of each sample. (float)\n\n"
      "  weights\n"
      "    Weights for the different data samples. (float)\n\n"
      "  arg_sorted_data\n"
      "    The argument used to sort data. (int32/64 dep. on architecture)\n\n"
      "Return::\n\n"
      "  min_value\n"
      "    The minimum weighted square error value result.\n\n"
      "  best_dim\n"
      "    The optimal dimension to use.\n\n"
      "  thr\n"
      "    The optimal threshold for this dimension.\n\n"
      "  coefficient\n"
      "    The optimal `a` value in the documentation.\n\n"
      "  constant\n"
      "    The optimal `b` value in the documentation.\n\n";

static PyObject *train_regression_stump(PyObject *self, PyObject *args)
{
    PyObject *data=NULL, *desired_output=NULL, *weights=NULL, *arg_sorted_data=NULL;
    PyObject *data_arr=NULL, *desired_output_arr=NULL, *weights_arr=NULL, *arg_sorted_arr=NULL;

    /* Local variables for handling the dimensions of the ndarrays. */
    long *data_arr_dims=NULL, *desired_output_arr_dims=NULL, *weights_arr_dims=NULL, *arg_sorted_arr_dims=NULL;
    int data_arr_nd=0, desired_output_arr_nd=0, weights_arr_nd=0, arg_sorted_arr_nd=0;

    /* Pointers into the ndarrays. */
    npy_double *data_ptr=NULL;
    npy_double *desired_output_ptr=NULL;
    npy_intp *arg_sorted_ptr=NULL;
    npy_double *weights_ptr=NULL;

    /* Stuff connected to the training */
    double tmp_val = 0.0;
    double current_weight, current_desired_output, coeff_a, coeff_b, current_err;
    double weights_sum, weights_do_sum;
    double weights_do_sum_end=0.0, weights_do_do_sum_end=0.0;
    double min_err_dim=1000000.0, min_err_global=1000000.0;
    double min_err_dim_coeff_a=0.0, min_err_dim_coeff_b=0.0, best_coeff_a=0.0, best_coeff_b=0.0;
    int min_err_dim_ind=0, best_dim=0;
    double thr=0.0;

    /* Variables. */
    int i, j;

    /* Parse the input to correct pointers. */
    if (!PyArg_ParseTuple(args, "OOOO", &data, &desired_output, &weights, &arg_sorted_data))
        return NULL;
    /* data_tuple is a tuple but now with a unknown number of arrays. */

    data_arr = PyArray_FROM_OTF(data, NPY_FLOAT64, NPY_IN_ARRAY);
    if (data_arr == NULL) goto fail;

    desired_output_arr = PyArray_FROM_OTF(desired_output, NPY_FLOAT64, NPY_IN_ARRAY);
    if (desired_output_arr == NULL) goto fail;

    weights_arr = PyArray_FROM_OTF(weights, NPY_FLOAT64, NPY_IN_ARRAY);
    if (weights_arr == NULL) goto fail;

    /* This will be an integer array but we want it 32/64 bit independent. */
    arg_sorted_arr = PyArray_FROM_O(arg_sorted_data);
    if (data_arr == NULL) goto fail;

    /* Dimensions of the arrays. */
    data_arr_nd                 = PyArray_NDIM(data_arr);
    data_arr_dims               = PyArray_DIMS(data_arr);
    desired_output_arr_nd       = PyArray_NDIM(desired_output_arr);
    desired_output_arr_dims     = PyArray_DIMS(desired_output_arr);
    weights_arr_nd              = PyArray_NDIM(weights_arr);
    weights_arr_dims            = PyArray_DIMS(weights_arr);
    arg_sorted_arr_nd           = PyArray_NDIM(arg_sorted_arr);
    arg_sorted_arr_dims         = PyArray_DIMS(arg_sorted_arr);

    if (data_arr_nd != 2)
    {
        printf("Dimension of data array shall be 2. (%d != 2).\n",
               data_arr_nd);
        goto fail;
    }

    if (desired_output_arr_nd != 1)
    {
        printf("Dimension of desired output array shall be 1. (%d != 1).\n",
               weights_arr_nd);
        goto fail;
    }

    if (weights_arr_nd != 1)
    {
        printf("Dimension of weights array shall be 1. (%d != 1).\n",
               weights_arr_nd);
        goto fail;
    }

    if (arg_sorted_arr_nd != 2)
    {
        printf("Dimension of arg sorted data shall be 2. (%d != 2).\n",
               data_arr_nd);
        goto fail;
    }

    if (data_arr_dims[1] != desired_output_arr_dims[0])
    {
        printf("Data and desired output must have the same length. (%ld != %ld).\n",
               data_arr_dims[1], desired_output_arr_dims[0]);
        goto fail;
    }

    if (data_arr_dims[1] != weights_arr_dims[0])
    {
        printf("Data and weights must have the same length. (%ld != %ld).\n",
               data_arr_dims[1], weights_arr_dims[0]);
        goto fail;
    }

    if ((data_arr_dims[0] != arg_sorted_arr_dims[0]) && (data_arr_dims[1] != arg_sorted_arr_dims[1]))
    {
        printf("Data and argsorted must be equal in shape. ((%ld, %ld) != (%ld, %ld)).\n",
               data_arr_dims[1], data_arr_dims[1], arg_sorted_arr_dims[1], arg_sorted_arr_dims[1]);
        goto fail;
    }

    /* We need get a pointer to the weights and the desired output.*/
    weights_ptr = (npy_double *)PyArray_GETPTR1(weights_arr, 0);
    desired_output_ptr = (npy_double *)PyArray_GETPTR1(desired_output_arr, 0);
    arg_sorted_ptr = (npy_intp *)PyArray_GETPTR2(arg_sorted_arr, 0, 0);

    // First, get the end sum of the weights multiplied by the desired_output.
    // The weights_do_do_sum_end is actually an unnecessary constant for the optimization,
    // but it is included (albeit applied first when returning results) for having correct
    // weighted square error value as output.
    tmp_val = 0.0;
    weights_do_sum_end = 0.0;
    weights_do_do_sum_end = 0.0;
    for(j = 0; j < data_arr_dims[1]; j++)
    {
        tmp_val = weights_ptr[arg_sorted_ptr[j]] * desired_output_ptr[arg_sorted_ptr[j]];
        weights_do_sum_end += tmp_val;
        weights_do_do_sum_end += tmp_val * desired_output_ptr[arg_sorted_ptr[j]];
    }

    /* Iterate over all features. */
    for(i = 0; i < data_arr_dims[0]; i++)
    {
        min_err_dim = 10000000.0;
        weights_sum = 0.0;
        weights_do_sum = 0.0;
        current_err = 0.0;

        arg_sorted_ptr = (npy_intp *)PyArray_GETPTR2(arg_sorted_arr, i, 0);

        /* Time to evaluate the result. */
        for(j = 0; j < data_arr_dims[1]; j++)
        {
            current_weight = weights_ptr[arg_sorted_ptr[j]];
            current_desired_output = desired_output_ptr[arg_sorted_ptr[j]];

            weights_sum += current_weight;
            weights_do_sum += current_weight * current_desired_output;

            coeff_b = weights_do_sum / weights_sum;
            if (weights_sum < 1.0)
            {
                coeff_a = ((weights_do_sum_end - weights_do_sum) / (1.0 - weights_sum)) - coeff_b;
            }
            else
            {
                coeff_a = (weights_do_sum_end - weights_do_sum) - coeff_b;
            }

            current_err = -(2 * coeff_a *(weights_do_sum_end - weights_do_sum)) -
                          (2 * coeff_b * weights_do_sum_end) +
                          (((coeff_a * coeff_a) + (2 * coeff_a * coeff_b)) * (1 - weights_sum)) +
                          (coeff_b * coeff_b);

            if(current_err < min_err_dim)
            {
                min_err_dim = current_err;
                min_err_dim_ind = j;
                min_err_dim_coeff_a = coeff_a;
                min_err_dim_coeff_b = coeff_b;
            }
        }

        /* Check if this is the best result so far. */
        if(min_err_dim < min_err_global)
        {
            min_err_global = min_err_dim;
            best_dim = i;
            best_coeff_a = min_err_dim_coeff_a;
            best_coeff_b = min_err_dim_coeff_b;

            /* The last point is the best. */
            if(min_err_dim_ind == (data_arr_dims[1] - 1))
            {
                arg_sorted_ptr = (npy_intp *)PyArray_GETPTR2(arg_sorted_arr, best_dim, min_err_dim_ind);
                data_ptr = (npy_double *)PyArray_GETPTR2(data_arr, best_dim, arg_sorted_ptr[0]);
                thr = data_ptr[0] + 1;
            }
            /* It is a special case if the min_point is in the beginning. */
            if(min_err_dim_ind == 0){
                arg_sorted_ptr = (npy_intp *)PyArray_GETPTR2(arg_sorted_arr, best_dim, min_err_dim_ind);
                data_ptr = (npy_double *)PyArray_GETPTR2(data_arr, best_dim, arg_sorted_ptr[0]);
                thr  = data_ptr[0] - 1;
            }
            else
            {
                /* TODO: Use get pointer less by using strides. This usage is to make it 32/64 independent. */
                arg_sorted_ptr = (npy_intp *)PyArray_GETPTR2(arg_sorted_arr, best_dim, min_err_dim_ind);
                data_ptr = (npy_double *)PyArray_GETPTR2(data_arr, best_dim, arg_sorted_ptr[0]);
                thr = data_ptr[0];
                arg_sorted_ptr = (npy_intp *)PyArray_GETPTR2(arg_sorted_arr, best_dim, (min_err_dim_ind + 1));
                data_ptr = (npy_double *)PyArray_GETPTR2(data_arr, best_dim, arg_sorted_ptr[0]);
                thr += data_ptr[0];
                thr /= 2;
            }
        }
    }

    /* Decrease reference counters to avoid memory leaks. */
    Py_DECREF(data_arr);
    Py_DECREF(desired_output_arr);
    Py_DECREF(weights_arr);
    Py_DECREF(arg_sorted_arr);

    min_err_global += weights_do_do_sum_end;
    return Py_BuildValue("diddd", min_err_global, best_dim, thr, best_coeff_a, best_coeff_b);

 fail:
    /* If error occurs, decrease reference counters to avoid memory leaks. */
    Py_XDECREF(data_arr);
    Py_XDECREF(desired_output_arr);
    Py_XDECREF(weights_arr);
    Py_XDECREF(arg_sorted_arr);

    /* Return error indication. */
    return NULL;

}


static PyMethodDef ClassifierMethods[] = {
    {"fit_regression_stump", train_regression_stump, METH_VARARGS, train_regression_stump_docs},
    {"fit_decision_stump", train_decision_stump, METH_VARARGS, train_decision_stump_docs},
     {NULL, NULL, 0, NULL} /* Sentinel */
};

PyMODINIT_FUNC initclassifiers(void)
{
    char * docstring = "Simple classifiers implemented in C.";
    (void) Py_InitModule3("classifiers", ClassifierMethods, docstring);
    import_array();
}
