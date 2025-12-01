#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for(size_t i = 0 ; i < m; i += batch) {
        size_t actual_batch_size = std::min(batch, m - i);
        // Initialize gradient matrix to zeros
        float* grad = new float[n * k]();  // () initializes to zero
        
        for(size_t j = 0; j < actual_batch_size; ++j) {
            // forward pass
            
            // compute logits
            float* logits = new float[k];
            for(size_t c = 0; c < k; ++c) {
                logits[c] = 0.0f;
                for(size_t d = 0; d < n; ++d) {
                    logits[c] += X[(i + j) * n + d] * theta[d * k + c];
                }
            }
            // Compute softmax probabilities
            float sum_exp = 0.0f;
            for(size_t c = 0; c < k; ++c) {
                logits[c] = std::exp(logits[c]);
                sum_exp += logits[c];
            }
            for(size_t c = 0; c < k; ++c) {
                logits[c] /= sum_exp;
            }
            // backward pass

            // Update gradient
            for(size_t d = 0; d < n; ++d) {
                for(size_t c = 0; c < k; ++c) {
                    float indicator = (y[i + j] == c) ? 1.0f : 0.0f;
                    grad[d * k + c] += (logits[c] - indicator) * X[(i + j) * n + d];
                }
            }
            delete[] logits;
        }
        // Update theta
        for(size_t d = 0; d < n; ++d) {
            for(size_t c = 0; c < k; ++c) {
                theta[d * k + c] -= (lr / actual_batch_size) * grad[d * k + c];
            }
        }
        delete[] grad;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
