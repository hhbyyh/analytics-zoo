/*
 * Copyright 2018 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <stdexcept>
#include <sstream>
#include <string>
#include <iostream>
#include <torch/script.h>
#include <memory>

#include "com_intel_analytics_zoo_pipeline_api_net_PytorchModel.h"
using namespace std;

extern "C" {

std::mutex mtx;
std::unordered_map<int, std::shared_ptr<torch::jit::script::Module>> handles;
std::unordered_map<int, std::shared_ptr<torch::jit::script::Module>> lossHandles;
std::unordered_map<int, at::Tensor> outputs;
long modelID;


JNIEXPORT jlong JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_loadNative
  (JNIEnv *jenv, jobject jobj, jstring jmodel_path, jstring jloss_path) {
    const char* p_model_path = jenv->GetStringUTFChars(jmodel_path, NULL);
    const char* p_loss_path = jenv->GetStringUTFChars(jloss_path, NULL);

    // Deserialize the ScriptModule from a file using torch::jit::load().
    std::shared_ptr<torch::jit::script::Module> model_ptr = torch::jit::load(p_model_path);
    std::shared_ptr<torch::jit::script::Module> loss_ptr = torch::jit::load(p_loss_path);
    assert(model_ptr != nullptr);
    assert(loss_ptr != nullptr);

    mtx.lock();
    modelID++;
    long id = modelID;
    handles.insert(std::make_pair(id, model_ptr));
    lossHandles.insert(std::make_pair(id, loss_ptr));
    mtx.unlock();

    jenv->ReleaseStringUTFChars(jmodel_path, p_model_path);
    jenv->ReleaseStringUTFChars(jloss_path, p_loss_path);
    return id;
  }



/*
 * Class:     com_intel_analytics_zoo_pipeline_api_net_PytorchModel
 * Method:    forward
 * Signature: ([F[I)Lcom/intel/analytics/zoo/pipeline/inference/JTensor;
 */
JNIEXPORT jobject JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_forwardNative
  (JNIEnv * jenv, jobject jobj, jlong nativeRef, jfloatArray jstorage, jint joffset, jintArray jshape) {

    // to Torch Tensor
    jfloat* c_storage = (jfloat*) jenv -> GetPrimitiveArrayCritical(jstorage, JNI_FALSE);
    jint* c_shape = (jint*) jenv -> GetPrimitiveArrayCritical(jshape, JNI_FALSE);
    int c_shape_len = jenv -> GetArrayLength(jshape);

    //Generate pytorch shape
    std::vector<int64_t> torch_shape;
    int i = 0;
    while(i < c_shape_len) {
        torch_shape.push_back(*(c_shape + i));
        i++;
    }
    // create a Tensor
    auto torch_input_tensor = torch::from_blob(c_storage + joffset, torch_shape, at::kFloat);

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch_input_tensor);

    std::shared_ptr<torch::jit::script::Module> model_ptr = handles[nativeRef];

    // Execute the model and turn its output into a tensor.
    at::Tensor output = model_ptr->forward(inputs).toTensor();
    mtx.lock();
    outputs.insert(std::make_pair(nativeRef, output));
    mtx.unlock();

    // Release critical part
    jenv -> ReleasePrimitiveArrayCritical(jstorage, c_storage, 0);
    jenv -> ReleasePrimitiveArrayCritical(jshape, c_shape, 0);

    // Wrap to Zoo JTensor
    jclass jtensor_class = jenv -> FindClass("com/intel/analytics/zoo/pipeline/inference/JTensor");
    jmethodID jtensor_constructor = jenv -> GetMethodID(jtensor_class, "<init>", "([F[I)V");

    auto sizes = output.sizes();

    int result_storage_len = 1;
    float *pytorch_result_storage = output.data<float>();
    int result_shape_len = sizes.size();

    int pytorch_result_shape[result_shape_len];
    int j = 0;
    while (j < result_shape_len) {
        pytorch_result_shape[j] = sizes[j];
        result_storage_len *= sizes[j];
        j++;
    }

    jfloatArray result_storage = jenv -> NewFloatArray(result_storage_len);
    jenv -> SetFloatArrayRegion(result_storage, 0, result_storage_len, pytorch_result_storage);

    jintArray result_shape = jenv -> NewIntArray(result_shape_len);
    jenv -> SetIntArrayRegion(result_shape, 0, result_shape_len, pytorch_result_shape);

    jobject result = jenv -> NewObject(jtensor_class, jtensor_constructor, result_storage, result_shape);

    return result;
  }


JNIEXPORT void JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_releaseNative
  (JNIEnv * jenv, jobject jobj, jlong nativeRef) {
    mtx.lock();
    handles.erase(nativeRef);
    mtx.unlock();
  }

JNIEXPORT jobject JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_backwardNative
  (JNIEnv * jenv, jobject jobj, jlong nativeRef, jfloatArray jstorage, jint joffset, jintArray jshape) {

    // to Torch Tensor
    jfloat* c_storage = (jfloat*) jenv -> GetPrimitiveArrayCritical(jstorage, JNI_FALSE);
    jint* c_shape = (jint*) jenv -> GetPrimitiveArrayCritical(jshape, JNI_FALSE);
    int c_shape_len = jenv -> GetArrayLength(jshape);

    //Generate pytorch shape
    std::vector<int64_t> torch_shape;
    int i = 0;
    while(i < c_shape_len) {
        torch_shape.push_back(*(c_shape + i));
        i++;
    }
    // create a Tensor
    auto label_tensor = torch::from_blob(c_storage + joffset, torch_shape, at::kFloat);

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> lossInputs;

    at::Tensor y = outputs[nativeRef];
    lossInputs.push_back(y);
    lossInputs.push_back(label_tensor);

    std::shared_ptr<torch::jit::script::Module> loss_ptr = lossHandles[nativeRef];
    std::cout << "lossInputs is: " << std::endl;
    std::cout << y << std::endl;
    std::cout << label_tensor << std::endl;
    at::Tensor loss = loss_ptr->forward(lossInputs).toTensor();
    std::cout << "loss is: " << std::endl;
    std::cout << loss << std::endl;
    loss.backward();

    // Release critical part
    jenv -> ReleasePrimitiveArrayCritical(jstorage, c_storage, 0);
    jenv -> ReleasePrimitiveArrayCritical(jshape, c_shape, 0);

    // Wrap to Zoo JTensor
    jclass jtensor_class = jenv -> FindClass("com/intel/analytics/zoo/pipeline/inference/JTensor");
    jmethodID jtensor_constructor = jenv -> GetMethodID(jtensor_class, "<init>", "([F[I)V");

    auto sizes = loss.sizes();

    int result_storage_len = 1;
    float *pytorch_result_storage = loss.data<float>();
    int result_shape_len = sizes.size();

    int pytorch_result_shape[result_shape_len];
    int j = 0;
    while (j < result_shape_len) {
        pytorch_result_shape[j] = sizes[j];
        result_storage_len *= sizes[j];
        j++;
    }

    jfloatArray result_storage = jenv -> NewFloatArray(result_storage_len);
    jenv -> SetFloatArrayRegion(result_storage, 0, result_storage_len, pytorch_result_storage);

    jintArray result_shape = jenv -> NewIntArray(result_shape_len);
    jenv -> SetIntArrayRegion(result_shape, 0, result_shape_len, pytorch_result_shape);

    jobject lossJTensor = jenv -> NewObject(jtensor_class, jtensor_constructor, result_storage, result_shape);

    return lossJTensor;

  }


JNIEXPORT jobject JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_getGradientNative
  (JNIEnv * jenv, jobject jobj, jlong nativeRef) {
    std::shared_ptr<torch::jit::script::Module> model_ptr = handles[nativeRef];
    std::vector<float> xv;

    for (const auto& child : model_ptr -> get_modules()) {

        auto slots = child -> get_parameters();
        for (size_t i = 0; i < slots.size(); ++i) {
             auto& x = slots[i];
             std::cout << "param: " << x.value().toTensor() << std::endl;
             std::cout << "grad: " << x.value().toTensor().grad() << std::endl;

             size_t x_size = x.value().toTensor().grad().numel();
             auto p = static_cast<float*>(x.value().toTensor().grad().storage().data());
             for(size_t i=0; i<x_size; i++)
             {
               xv.push_back(p[i]);
             }
        }
    }

    std::cout << "xv: " << xv << std::endl;
    jclass jtensor_class = jenv -> FindClass("com/intel/analytics/zoo/pipeline/inference/JTensor");
    jmethodID jtensor_constructor = jenv -> GetMethodID(jtensor_class, "<init>", "([F[I)V");

    int result_storage_len = xv.size();
    jfloatArray result_storage = jenv -> NewFloatArray(result_storage_len);
    jenv -> SetFloatArrayRegion(result_storage, 0, result_storage_len, &xv[0]);

    int pytorch_result_shape[1] = {xv.size()};
    jintArray result_shape = jenv -> NewIntArray(1);
    jenv -> SetIntArrayRegion(result_shape, 0, 1, pytorch_result_shape);

    jobject gradientJTensor = jenv -> NewObject(jtensor_class, jtensor_constructor, result_storage, result_shape);

    return gradientJTensor;
  }


JNIEXPORT jobject JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_updateWeightNative
  (JNIEnv * jenv, jobject jobj, jlong nativeRef, jfloatArray jstorage) {
    std::shared_ptr<torch::jit::script::Module> model_ptr = handles[nativeRef];
    std::vector<float> xv;

    for (const auto& child : model_ptr -> get_modules()) {

        auto slots = child -> get_parameters();
        for (size_t i = 0; i < slots.size(); ++i) {
             auto& x = slots[i];
             std::cout << "param: " << x.value().toTensor() << std::endl;
             std::cout << "grad: " << x.value().toTensor().grad() << std::endl;

             size_t x_size = x.value().toTensor().grad().numel();
             auto p = static_cast<float*>(x.value().toTensor().grad().storage().data());
             for(size_t i=0; i<x_size; i++)
             {
               xv.push_back(p[i]);
             }
        }
    }
  }

}
