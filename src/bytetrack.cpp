#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <dirent.h>
#include <sys/time.h>
#include "BYTETracker.h"
#include "rknn_api.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/video.hpp"


#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            cerr << "Cuda failure: " << ret << endl;\
            abort();\
        }\
    } while (0)

#define NMS_THRESH 0.5                       // NMS阈值
#define NUM_ANCHORS 3                        // anchor的数量
#define NUM_CLASSES 2                        // 类别数
#define BBOX_CONF_THRESH 0.1                 // 置信度阈值
#define OBJ_NUMB_MAX_SIZE 64                 // 最多检测的目标数
#define SAVE_PATH "output.avi"

static const int INPUT_W = 640;              // 模型输入width
static const int INPUT_H = 640;              // 模型输入height
const int anchor[3][6] = {10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};
const string label_name[2] = {"person","vehicle"};


cv::Mat static_resize(cv::Mat& img) {
    float r = min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    // r = std::min(r, 1.0f);
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(Rect(0, 0, re.cols, re.rows)));
    return out;
}

struct GridAndStride
{
    int grid0; // grid_w：模型输入width/stride   width方向的网格数量
    int grid1; // grid_h：模型输入height/stride    height方向的网格数量
    int stride; // 下采样倍数，取值有8、16、32
};

static void check_ret(int ret, string ret_name)
{
    // 检查ret是否正确并输出，ret_name表示哪一步
    if (ret < 0)
    {
        cout << ret_name << " error ret=" << ret << endl;
    }
}

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    // 打印模型输入和输出的信息
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

// t为结构体，存储了时间信息：1、tv_sec 代表多少秒；2、tv_usec 代表多少微秒， 1000000 微秒 = 1秒
double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec);} 

static float sigmoid(float x)
{
    return 1.0 / (1.0 + expf(-x));
}

static float unsigmoid(float y)
{
    return -1.0 * logf((1.0 / y) - 1.0);
}

inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)  // 量化
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale)  // 反量化
{
    return ((float)qnt - (float)zp) * scale;
}

static void generate_grids_and_stride(const int target_w, const int target_h, vector<int>& strides, vector<GridAndStride>& grid_strides)
{
    /*
        生成网格(直接精确到具体哪个网格)和步幅[没用到]
        target_w：模型输入width
        target_h：模型输入height
        strides：下采样倍数  vector<int> strides = {8, 16, 32};
        grid_stride：vector<GridAndStride> grid_strides;上面定义了GridAndStride结构体
    */
    for (auto stride : strides)
    {
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                // printf("generate_grids_and_stride:  %d\t,%d\t,%d\n", g0, g1, stride);
                grid_strides.push_back((GridAndStride){g0, g1, stride});
            }
        }
    }
}
static void generate_grids(const int target_w, const int target_h, vector<int>& strides, vector<GridAndStride>& grid_strides)
{
    /*
        生成网格和步幅
        target_w：模型输入width
        target_h：模型输入height
        strides：下采样倍数  vector<int> strides = {8, 16, 32};
        grid_stride：vector<GridAndStride> grid_strides;上面定义了GridAndStride结构体
    */
    for (auto stride : strides)
    {
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        grid_strides.push_back((GridAndStride){num_grid_w, num_grid_h, stride});
        
    }
}

static inline float intersection_area(const Object& a, const Object& b)
{
    Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const vector<Object>& faceobjects, vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static void generate_yolo_proposals(GridAndStride grid_stride, int8_t *output, int *anchor, float prob_threshold, vector<Object>& objects, int32_t zp, float scale)
{
    const int num_class = NUM_CLASSES;
    const int num_anchors = NUM_ANCHORS;
    const int grid0 = grid_stride.grid0;
    const int grid1 = grid_stride.grid1;
    const int stride = grid_stride.stride;
    int grid_len = grid0 * grid1;
    float thres = unsigmoid(prob_threshold);
    int8_t thres_i8 = qnt_f32_to_affine(thres, zp, scale);

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        // 三个anchor
        for (int i = 0; i < grid1; i++)
        {
            for (int j = 0; j < grid0; j++)
            {
                int8_t box_objectness = output[(anchor_idx * (num_class + 5) + 4) * grid_len + i * grid0 + j];
                if (box_objectness >= thres_i8)
                {
                    const int basic_pos = anchor_idx * (num_class + 5) * grid_len + i * grid0 + j;; // 相当于detection.py中的(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j;
                    int8_t *in_ptr = output + basic_pos;
                    float box_x = sigmoid(deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
                    float box_y = sigmoid(deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
                    float box_w = sigmoid(deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
                    float box_h = sigmoid(deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
                    float x_center = (box_x + j) * (float)stride;
                    float y_center = (box_y + i) * (float)stride;
                    float w = box_w * box_w * (float)anchor[anchor_idx * 2];
                    float h = box_h * box_h * (float)anchor[anchor_idx * 2 + 1];
                    float x0 = x_center - w * 0.5f;
                    float y0 = y_center - h * 0.5f;

                    for (int class_idx = 0; class_idx < num_class; class_idx++)
                    {
                        int8_t box_cls_score = output[(anchor_idx * (num_class + 5) + 5) * grid_len + i * grid0 + j + class_idx];
                        Object obj;
                        obj.rect.x = x0;
                        obj.rect.y = y0;
                        obj.rect.width = w;
                        obj.rect.height = h;
                        obj.label = class_idx;
                        obj.prob = sigmoid(deqnt_affine_to_f32(box_cls_score, zp, scale));

                        objects.push_back(obj);
                    }
                }
            }
        }
    }
}
float* blobFromImage(cv::Mat& img){
    // 归一化操作 将原始图像拉成一维数组了
    cvtColor(img, img, COLOR_BGR2RGB);

    float* blob = new float[img.total()*3]; //Mat.total()返回Mat的单个通道中的元素总数，该元素等于Mat.size中值的乘积。对于RGB图像，total() = rows*cols。
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    vector<float> mean = {0.485, 0.456, 0.406};
    vector<float> std = {0.229, 0.224, 0.225};
    for (size_t c = 0; c < channels; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    (((float)img.at<Vec3b>(h, w)[c]) / 255.0f - mean[c]) / std[c];
            }
        }
    }
    return blob;
}


static void decode_outputs(rknn_output *outputs, vector<Object>& objects, float resize_scale, const int img_w, const int img_h, int32_t *qnt_zps, float *qnt_scales) {
        vector<Object> proposals;
        vector<int> strides = {8, 16, 32};
        vector<GridAndStride> grid_strides;
        generate_grids(INPUT_W, INPUT_H, strides, grid_strides);
        for(int i = 0; i < 3; i++){
            // 三个尺度
            generate_yolo_proposals(grid_strides[i], (int8_t *)outputs[i].buf, (int *)anchor[i], BBOX_CONF_THRESH, proposals, qnt_zps[i], qnt_scales[i]);
        }
        std::cout << "num of boxes before nms: " << proposals.size() << std::endl;

        qsort_descent_inplace(proposals);

        vector<int> picked;
        nms_sorted_bboxes(proposals, picked, NMS_THRESH);


        int count = picked.size();

        std::cout << "num of boxes: " << count << std::endl;

        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (objects[i].rect.x) / resize_scale;
            float y0 = (objects[i].rect.y) / resize_scale;
            float x1 = (objects[i].rect.x + objects[i].rect.width) / resize_scale;
            float y1 = (objects[i].rect.y + objects[i].rect.height) / resize_scale;

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;
        }
}

const float color_list[80][3] =
{
    {0.000, 0.447, 0.741},
    {0.850, 0.325, 0.098},
    {0.929, 0.694, 0.125},
    {0.494, 0.184, 0.556},
    {0.466, 0.674, 0.188},
    {0.301, 0.745, 0.933},
    {0.635, 0.078, 0.184},
    {0.300, 0.300, 0.300},
    {0.600, 0.600, 0.600},
    {1.000, 0.000, 0.000},
    {1.000, 0.500, 0.000},
    {0.749, 0.749, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.333, 0.333, 0.000},
    {0.333, 0.667, 0.000},
    {0.333, 1.000, 0.000},
    {0.667, 0.333, 0.000},
    {0.667, 0.667, 0.000},
    {0.667, 1.000, 0.000},
    {1.000, 0.333, 0.000},
    {1.000, 0.667, 0.000},
    {1.000, 1.000, 0.000},
    {0.000, 0.333, 0.500},
    {0.000, 0.667, 0.500},
    {0.000, 1.000, 0.500},
    {0.333, 0.000, 0.500},
    {0.333, 0.333, 0.500},
    {0.333, 0.667, 0.500},
    {0.333, 1.000, 0.500},
    {0.667, 0.000, 0.500},
    {0.667, 0.333, 0.500},
    {0.667, 0.667, 0.500},
    {0.667, 1.000, 0.500},
    {1.000, 0.000, 0.500},
    {1.000, 0.333, 0.500},
    {1.000, 0.667, 0.500},
    {1.000, 1.000, 0.500},
    {0.000, 0.333, 1.000},
    {0.000, 0.667, 1.000},
    {0.000, 1.000, 1.000},
    {0.333, 0.000, 1.000},
    {0.333, 0.333, 1.000},
    {0.333, 0.667, 1.000},
    {0.333, 1.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.667, 0.333, 1.000},
    {0.667, 0.667, 1.000},
    {0.667, 1.000, 1.000},
    {1.000, 0.000, 1.000},
    {1.000, 0.333, 1.000},
    {1.000, 0.667, 1.000},
    {0.333, 0.000, 0.000},
    {0.500, 0.000, 0.000},
    {0.667, 0.000, 0.000},
    {0.833, 0.000, 0.000},
    {1.000, 0.000, 0.000},
    {0.000, 0.167, 0.000},
    {0.000, 0.333, 0.000},
    {0.000, 0.500, 0.000},
    {0.000, 0.667, 0.000},
    {0.000, 0.833, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 0.167},
    {0.000, 0.000, 0.333},
    {0.000, 0.000, 0.500},
    {0.000, 0.000, 0.667},
    {0.000, 0.000, 0.833},
    {0.000, 0.000, 1.000},
    {0.000, 0.000, 0.000},
    {0.143, 0.143, 0.143},
    {0.286, 0.286, 0.286},
    {0.429, 0.429, 0.429},
    {0.571, 0.571, 0.571},
    {0.714, 0.714, 0.714},
    {0.857, 0.857, 0.857},
    {0.000, 0.447, 0.741},
    {0.314, 0.717, 0.741},
    {0.50, 0.5, 0}
};

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    /* 
    加载rknn模型
    filename : rknn模型文件路径
    model_size : 模型的大小
    */
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open rknn model file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

int main(int argc, char** argv) 
{
    struct timeval start_time, stop_time, begin_time, end_time; // 用于计时
    const char *model_name = argv[1];
    // 开始计时
    gettimeofday(&begin_time, NULL);
    /********************rknn init*********************/
    string ret_name;
    ret_name = "rknn_init"; // 表示rknn的步骤名称
    rknn_context ctx; // 创建rknn_context对象
    int model_data_size = 0; // 模型的大小
    unsigned char *model_data = load_model(model_name, &model_data_size); // 加载RKNN模型
    /* 初始化参数flag
    RKNN_FLAG_COLLECT_PERF_MASK：用于运行时查询网络各层时间。
    RKNN_FLAG_MEM_ALLOC_OUTSIDE：用于表示模型输入、输出、权重、中间 tensor 内存全部由用户分配。
    */
    int ret = rknn_init(&ctx, model_data, model_data_size, RKNN_FLAG_COLLECT_PERF_MASK, NULL); // 初始化RKNN
    check_ret(ret, ret_name);
    // 设置NPU核心为自动调度
    rknn_core_mask core_mask = RKNN_NPU_CORE_AUTO;
    ret = rknn_set_core_mask(ctx, core_mask);
    

    /********************rknn query*********************/
    // rknn_query 函数能够查询获取到模型输入输出信息、逐层运行时间、模型推理的总时间、
    // SDK 版本、内存占用信息、用户自定义字符串等信息。
    // 版本信息
    ret_name = "rknn_query";
    rknn_sdk_version version; // SDK版本信息结构体
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    check_ret(ret, ret_name);
    printf("sdk api version: %s\n", version.api_version);
    printf("driver version: %s\n", version.drv_version);

    // 输入输出信息
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    check_ret(ret, ret_name);
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // 输入输出Tensor属性
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs)); // 初始化内存
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i; // 输入的索引位置
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        check_ret(ret, ret_name);
        dump_tensor_attr(&(input_attrs[i]));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        check_ret(ret, ret_name);
        dump_tensor_attr(&(output_attrs[i]));
    }

    // 模型输入信息
    int channel = 3;
    int width = 0;
    int height = 0;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        channel = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
        height = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        width = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }

    
    const string input_video_path {argv[2]};
    static float* prob = new float[OBJ_NUMB_MAX_SIZE]; //

    //cv::VideoCapture cap(input_video_path);
    cv::VideoCapture cap;
    cap.open(input_video_path);
	if (!cap.isOpened())
		return 0;

	int img_w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int img_h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(cv::CAP_PROP_FPS);
    long nFrame = static_cast<long>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    cout << "Total frames: " << nFrame << endl;

    // cv::VideoWriter writer("demo.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(img_w, img_h));
    cv::VideoWriter writer;
    writer = cv::VideoWriter(SAVE_PATH, cv::VideoWriter::fourcc('M','P','E','G'), fps, cv::Size(img_w, img_h));
    cv::Mat img;
    BYTETracker tracker(fps, 50);
    int num_frames = 0;
    int total_ms = 0;
   
	while (true)
    {
        if(!cap.read(img))
        {
            cout <<" Break at cap.read(img) !" << endl;
            break;
        }
        num_frames ++;
        cout << "Frames index:" << num_frames << endl;
        if (num_frames % 20 == 0)
        {
            cout << "Processing frame " << num_frames << " (" << num_frames * 1000000 / total_ms << " fps)" << endl;
        }
		if (img.empty())
        {
			break;
        }
        cv::Mat pr_img = static_resize(img); // resize图像
        // float* blob;         
        // blob = blobFromImage(pr_img); // 归一化图像
        float resize_scale = min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0)); // 图像缩放尺度
        
        /********************rknn inputs set*********************/
        gettimeofday(&start_time, NULL);
        ret_name = "rknn_inputs_set";
        rknn_input inputs[1];
        memset(inputs, 0, sizeof(inputs));
    
        inputs[0].index = 0;                       // 输入的索引位置
        inputs[0].type = RKNN_TENSOR_UINT8;        // 输入数据类型 采用INT8
        inputs[0].size = width * height * channel; // 这里用的是模型的
        inputs[0].fmt = input_attrs[0].fmt;        // 输入格式，NHWC
        inputs[0].pass_through = 0;                // 为0代表需要进行预处理
        inputs[0].buf = pr_img.data;               // 如果进行resize需要改为resize的data

        ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
        check_ret(ret, ret_name);
        gettimeofday(&stop_time, NULL);
        // printf("rknn input %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);
        /********************rknn run****************************/
        gettimeofday(&start_time, NULL);
        ret_name = "rknn_run";
        ret = rknn_run(ctx, NULL); // 推理
        check_ret(ret, ret_name);
        gettimeofday(&stop_time, NULL);
        total_ms = total_ms + (__get_us(stop_time) - __get_us(start_time)) / 1000;
        printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

        /********************rknn outputs get****************************/
        gettimeofday(&start_time, NULL);
        ret_name = "rknn_outputs_get";
        float out_scales[3] = {0, 0, 0}; // 存储scales 和 zp
        int32_t out_zps[3] = {0, 0, 0};
        // 创建rknn_output对象
        rknn_output outputs[io_num.n_output];
        memset(outputs, 0, sizeof(outputs));
        for (int i = 0; i < io_num.n_output; i++) 
        { 
            outputs[i].index = i; // 输出索引
            outputs[i].is_prealloc = 0; // 由rknn来分配输出的buf，指向输出数据
            outputs[i].want_float = 0;
            out_scales[i] = output_attrs[i].scale;
            out_zps[i] = output_attrs[i].zp; 
        }
        ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
        gettimeofday(&stop_time, NULL);
        // printf("rknn output %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);
        /********************是否打印推理时间细节****************************/
        ret_name = "rknn_perf_detail_display";
        rknn_perf_detail perf_detail;
        ret = rknn_query(ctx, RKNN_QUERY_PERF_DETAIL, &perf_detail, sizeof(perf_detail));
        check_ret(ret, ret_name);
        // printf("%s\n",perf_detail.perf_data);

        // 解码推理结果并进行跟踪
        vector<Object> objects;
        decode_outputs(outputs, objects, resize_scale, img_w, img_h, out_zps, out_scales);
        vector<STrack> output_stracks = tracker.update(objects);

        // 画检测的框
        /*
        for (int i = 0; i < objects.size(); i++)
        {
            // objects[i].rect.x = x0;
            // objects[i].rect.y = y0;
            // objects[i].rect.width = x1 - x0;
            // objects[i].rect.height = y1 - y0;
            

            cv::Scalar s = tracker.get_color(1);
		    putText(img, label_name[objects[i].label], Point(objects[i].rect.x, objects[i].rect.y - 5), 
                    0, 0.6, cv::Scalar(0, 0, 255), 2, LINE_AA);
            rectangle(img, Rect(objects[i].rect.x, objects[i].rect.y, objects[i].rect.width, objects[i].rect.height), s, 2);
        }
        */

        // 画跟踪的框
        
        for (int i = 0; i < output_stracks.size(); i++)
		{
			vector<float> tlwh = output_stracks[i].tlwh;
			cv::Scalar s = tracker.get_color(output_stracks[i].track_id);
			putText(img, format("%d", output_stracks[i].track_id), Point(tlwh[0], tlwh[1] - 5), 
                    0, 0.6, cv::Scalar(0, 0, 255), 2, LINE_AA);
            rectangle(img, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
			
		}
        /*
        for (int i = 0; i < output_stracks.size(); i++)
		{
			vector<float> tlwh = output_stracks[i].tlwh;
			bool vertical = tlwh[2] / tlwh[3] > 1.6;
			if (tlwh[2] * tlwh[3] > 20 && !vertical)
			{
				cv::Scalar s = tracker.get_color(output_stracks[i].track_id);
				putText(img, format("%d", output_stracks[i].track_id), Point(tlwh[0], tlwh[1] - 5), 
                        0, 0.6, cv::Scalar(0, 0, 255), 2, LINE_AA);
                rectangle(img, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
			}
		}
        */
        putText(img, format("frame: %d fps: %d num: %ld", num_frames, num_frames * 1000 / total_ms  , output_stracks.size()), 
                Point(0, 30), 0, 0.6, cv::Scalar(0, 0, 255), 2, LINE_AA);
        writer.write(img);
    }
    // 结束计时
    gettimeofday(&end_time, NULL);
    cap.release();
    printf("all time: %f ms\n", (__get_us(end_time) - __get_us(begin_time)) / 1000);
    cout << "FPS: " << num_frames * 1000 / total_ms << endl;
    return 0;
}
