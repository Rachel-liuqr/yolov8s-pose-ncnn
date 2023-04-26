#include "layer.h"
#include "net.h"

#include "opencv2/opencv.hpp"

#include <float.h>
#include <stdio.h>
#include <vector>

#define MAX_STRIDE 32 // if yolov8-p6 model modify to 64


const std::vector<std::vector<unsigned int>> KPS_COLORS =
        {{0,   255, 0},
         {0,   255, 0},
         {0,   255, 0},
         {0,   255, 0},
         {0,   255, 0},
         {255, 128, 0},
         {255, 128, 0},
         {255, 128, 0},
         {255, 128, 0},
         {255, 128, 0},
         {255, 128, 0},
         {51,  153, 255},
         {51,  153, 255},
         {51,  153, 255},
         {51,  153, 255},
         {51,  153, 255},
         {51,  153, 255}};

const std::vector<std::vector<unsigned int>> SKELETON = {{16, 14},
                                                         {14, 12},
                                                         {17, 15},
                                                         {15, 13},
                                                         {12, 13},
                                                         {6,  12},
                                                         {7,  13},
                                                         {6,  7},
                                                         {6,  8},
                                                         {7,  9},
                                                         {8,  10},
                                                         {9,  11},
                                                         {2,  3},
                                                         {1,  2},
                                                         {1,  3},
                                                         {2,  4},
                                                         {3,  5},
                                                         {4,  6},
                                                         {5,  7}};

const std::vector<std::vector<unsigned int>> LIMB_COLORS = {{51,  153, 255},
                                                            {51,  153, 255},
                                                            {51,  153, 255},
                                                            {51,  153, 255},
                                                            {255, 51,  255},
                                                            {255, 51,  255},
                                                            {255, 51,  255},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {0,   255, 0},
                                                            {0,   255, 0},
                                                            {0,   255, 0},
                                                            {0,   255, 0},
                                                            {0,   255, 0},
                                                            {0,   255, 0},
                                                            {0,   255, 0}};

struct Object
{
	cv::Rect_<float> rect;
	int label;
	float prob;
	std::vector<float> kps;
};

static float sigmod(const float in)
{
	return 1.f / (1.f + expf(-1.f * in));
}

static float softmax(
	const float* src,
	float* dst,
	int length
)
{
	float alpha = -FLT_MAX;
	for (int c = 0; c < length; c++)
	{
		float score = src[c];
		if (score > alpha)
		{
			alpha = score;
		}
	}

	float denominator = 0;
	float dis_sum = 0;
	for (int i = 0; i < length; ++i)
	{
		dst[i] = expf(src[i] - alpha);
		denominator += dst[i];
	}
	for (int i = 0; i < length; ++i)
	{
		dst[i] /= denominator;
		dis_sum += i * dst[i];
	}
	return dis_sum;
}

static void generate_proposals(
	int stride,
	const ncnn::Mat& feat_blob,
	const float prob_threshold,
	std::vector<Object>& objects
)
{
	const int reg_max = 16;
	float dst[16];
	const int num_w = feat_blob.w;
	const int num_grid_y = feat_blob.c;
	const int num_grid_x = feat_blob.h;
	const int kps_num = 17;

	const int num_class = num_w - 4 * reg_max;

	const int clsid = 0;

	for (int i = 0; i < num_grid_y; i++)
	{
		for (int j = 0; j < num_grid_x; j++)
		{
			const float* matat = feat_blob.channel(i).row(j);

			float score = matat[0];
            score = sigmod(score);
			if (score < prob_threshold)
			{
				continue;
			}

			float x0 = j + 0.5f - softmax(matat + 1, dst, 16);
			float y0 = i + 0.5f - softmax(matat + (1 + 16), dst, 16);
			float x1 = j + 0.5f + softmax(matat + (1 + 2 * 16), dst, 16);
			float y1 = i + 0.5f + softmax(matat + (1 + 3 * 16), dst, 16);

			x0 *= stride;
			y0 *= stride;
			x1 *= stride;
			y1 *= stride;

			std::vector<float> kps;
			for(int k=0; k<kps_num; k++)
			{
                float kps_x = (matat[1 + 64 + k * 3] * 2.f+ j) * stride;
                float kps_y = (matat[1 + 64 + k * 3 + 1] * 2.f + i) * stride;
                float kps_s = sigmod(matat[1 + 64 + k * 3 + 2]);
				
				kps.push_back(kps_x);
                kps.push_back(kps_y);
                kps.push_back(kps_s);
			}

			Object obj;
			obj.rect.x = x0;
			obj.rect.y = y0;
			obj.rect.width = x1 - x0;
			obj.rect.height = y1 - y0;
			obj.label = 0;
			obj.prob = score;
			obj.kps = kps;
			objects.push_back(obj);
		}
	}

}

static float clamp(
	float val,
	float min = 0.f,
	float max = 1280.f
)
{
	return val > min ? (val < max ? val : max) : min;
}

static void non_max_suppression(
	std::vector<Object>& proposals,
	std::vector<Object>& results,
	int orin_h,
	int orin_w,
	float dh = 0,
	float dw = 0,
	float ratio_h = 1.0f,
	float ratio_w = 1.0f,
	float conf_thres = 0.25f,
	float iou_thres = 0.65f
)
{
	results.clear();
	std::vector<cv::Rect> bboxes;
	std::vector<float> scores;
	std::vector<int> labels;
	std::vector<int> indices;
    std::vector<std::vector<float>> kpss;

	for (auto& pro : proposals)
	{
		bboxes.push_back(pro.rect);
		scores.push_back(pro.prob);
		labels.push_back(pro.label);
        kpss.push_back(pro.kps);
	}

	cv::dnn::NMSBoxes(
		bboxes,
		scores,
		conf_thres,
		iou_thres,
		indices
	);

	for (auto i : indices)
	{
		auto& bbox = bboxes[i];
		float x0 = bbox.x;
		float y0 = bbox.y;
		float x1 = bbox.x + bbox.width;
		float y1 = bbox.y + bbox.height;
		float& score = scores[i];
		int& label = labels[i];

		x0 = (x0 - dw) / ratio_w;
		y0 = (y0 - dh) / ratio_h;
		x1 = (x1 - dw) / ratio_w;
		y1 = (y1 - dh) / ratio_h;

		x0 = clamp(x0, 0.f, orin_w);
		y0 = clamp(y0, 0.f, orin_h);
		x1 = clamp(x1, 0.f, orin_w);
		y1 = clamp(y1, 0.f, orin_h);

		Object obj;
		obj.rect.x = x0;
		obj.rect.y = y0;
		obj.rect.width = x1 - x0;
		obj.rect.height = y1 - y0;
		obj.prob = score;
		obj.label = label;
        obj.kps = kpss[i];
		for(int n=0; n<obj.kps.size(); n+=3)
		{
			obj.kps[n] = clamp((obj.kps[n] - dw) / ratio_w, 0.f, orin_w);
			obj.kps[n + 1] = clamp((obj.kps[n + 1] - dh) / ratio_h, 0.f, orin_h);
		}

		results.push_back(obj);
	}
}

static int detect_yolov8(const cv::Mat& bgr, std::vector<Object>& objects)
{
	ncnn::Net yolov8;

	yolov8.opt.use_vulkan_compute = true;
	// yolov8.opt.use_bf16_storage = true;

	// original pretrained model from https://github.com/ultralytics/ultralytics
	// the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    std::string paramPath = "yolov8s-pose-opt-fp16.param";
    std::string binPath = "yolov8s-pose-opt-fp16.bin";
	std::cout<<paramPath<<std::endl<<binPath<<std::endl;

	if (yolov8.load_param(paramPath.c_str()))
		exit(-1);
	if (yolov8.load_model(binPath.c_str()))
		exit(-1);

	const int target_size = 640;
	const float prob_threshold = 0.25f;
	const float nms_threshold = 0.45f;

	int img_w = bgr.cols;
	int img_h = bgr.rows;

	// letterbox pad to multiple of MAX_STRIDE
	int w = img_w;
	int h = img_h;
	float scale = 1.f;
	if (w > h)
	{
		scale = (float)target_size / w;
		w = target_size;
		h = h * scale;
	}
	else
	{
		scale = (float)target_size / h;
		h = target_size;
		w = w * scale;
	}

	ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

	// pad to target_size rectangle
	// ultralytics/yolo/data/dataloaders/v5augmentations.py letterbox
	int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
	int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;

	int top = hpad / 2;
	int bottom = hpad - hpad / 2;
	int left = wpad / 2;
	int right = wpad - wpad / 2;

	ncnn::Mat in_pad;
	ncnn::copy_make_border(in,
		in_pad,
		top,
		bottom,
		left,
		right,
		ncnn::BORDER_CONSTANT,
		114.f);

	const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
	in_pad.substract_mean_normalize(0, norm_vals);

	ncnn::Extractor ex = yolov8.create_extractor();

	ex.input("images", in_pad);

	std::vector<Object> proposals;

	// stride 8
	{
		ncnn::Mat out;
		ex.extract("356", out);

		std::vector<Object> objects8;
		generate_proposals(8, out, prob_threshold, objects8);

		proposals.insert(proposals.end(), objects8.begin(), objects8.end());
	}

	// stride 16
	{
		ncnn::Mat out;
		ex.extract("381", out);

		std::vector<Object> objects16;
		generate_proposals(16, out, prob_threshold, objects16);

		proposals.insert(proposals.end(), objects16.begin(), objects16.end());
	}

	// stride 32
	{
		ncnn::Mat out;
		ex.extract("406", out);

		std::vector<Object> objects32;
		generate_proposals(32, out, prob_threshold, objects32);

		proposals.insert(proposals.end(), objects32.begin(), objects32.end());
	}

	non_max_suppression(proposals, objects,
		img_h, img_w, hpad / 2, wpad / 2,
		scale, scale, prob_threshold, nms_threshold);
	return 0;
}

void draw_objects(
        const cv::Mat &image,
        cv::Mat &res,
        const std::vector<Object> &objs,
        const std::vector<std::vector<unsigned int>> &SKELETON,
        const std::vector<std::vector<unsigned int>> &KPS_COLORS,
        const std::vector<std::vector<unsigned int>> &LIMB_COLORS
) {
    res = image.clone();
    const int num_point = 17;
    for (auto &obj: objs) {
        cv::rectangle(
                res,
                obj.rect,
                {0, 0, 255},
                2
        );

        char text[256];
        sprintf(
                text,
                "person %.1f%%",
                obj.prob * 100
        );

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(
                text,
                cv::FONT_HERSHEY_SIMPLEX,
                0.4,
                1,
                &baseLine
        );

        int x = (int) obj.rect.x;
        int y = (int) obj.rect.y + 1;

        if (y > res.rows)
            y = res.rows;

        cv::rectangle(
                res,
                cv::Rect(x, y, label_size.width, label_size.height + baseLine),
                {0, 0, 255},
                -1
        );

        cv::putText(
                res,
                text,
                cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX,
                0.4,
                {255, 255, 255},
                1
        );

        auto &kps = obj.kps;
        for (int k = 0; k < num_point + 2; k++)
		{
            if (k < num_point)
			{
                int kps_x = std::round(kps[k * 3]);
                int kps_y = std::round(kps[k * 3 + 1]);
                float kps_s = kps[k * 3 + 2];
                if (kps_s > 0.5f)
				{
                    cv::Scalar kps_color = cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
                    cv::circle(res, {kps_x, kps_y}, 5, kps_color, -1);
                }
            }
            auto &ske = SKELETON[k];
            int pos1_x = std::round(kps[(ske[0] - 1) * 3]);
            int pos1_y = std::round(kps[(ske[0] - 1) * 3 + 1]);

            int pos2_x = std::round(kps[(ske[1] - 1) * 3]);
            int pos2_y = std::round(kps[(ske[1] - 1) * 3 + 1]);

            float pos1_s = kps[(ske[0] - 1) * 3 + 2];
            float pos2_s = kps[(ske[1] - 1) * 3 + 2];

            if (pos1_s > 0.5f && pos2_s > 0.5f)
			{
                cv::Scalar limb_color = cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
                cv::line(res, {pos1_x, pos1_y}, {pos2_x, pos2_y}, limb_color, 2);
            }
        }
    }
}


int main(int argc, char** argv)
{
	if (argc != 2)
	{
		fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
		return -1;
	}

	const char* imagepath = argv[1];

	cv::Mat m = cv::imread(imagepath, 1);
	if (m.empty())
	{
		fprintf(stderr, "cv::imread %s failed\n", imagepath);
		return -1;
	}

	std::vector<Object> objects;
	detect_yolov8(m, objects);

	cv::Mat res;
	draw_objects(m, res, objects, SKELETON, KPS_COLORS, LIMB_COLORS);

	cv::imshow("result", res);
	cv::waitKey(0);

	return 0;
}
