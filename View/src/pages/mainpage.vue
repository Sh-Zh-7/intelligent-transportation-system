<template>
  <div id="mainpage">
    <div class="title front">智能交通</div>
    <div class="title rear">连接未来</div>
    <div class="subtitle">基于计算机视觉的交通场景智能应用</div>
    <div class="uploadcomponent">
      <div class="uploadvideo" :class="videoStyle">
        <ele-upload-video
          :fileSize="20"
          :responseFn="handleResponse"
          action="http://127.0.0.1:8000/upload/video"
          name="video"
          :value="video"
        />
        <el-tag class="infotag" v-if="!VideoUploadSuccess" type="info" effect="dark">请上传待处理视频</el-tag>
        <el-tag class="infotag" v-if="VideoUploadSuccess" type="success" effect="dark">视频上传成功</el-tag>
      </div>
      <div class="uploadimage" :class="imageStyle">
        <div id="ele-upload-img">
          <el-upload
            class="avatar-uploader"
            action="http://127.0.0.1:8000/upload/image"
            :show-file-list="false"
            :on-success="handleUploadSuccess"
            :on-error="handleUploadError"
            name="image"
          >
            <img v-if="imageUrl" :src="imageUrl" class="avatar" />
            <i v-else class="el-icon-plus avatar-uploader-icon"></i>
          </el-upload>
        </div>
        <el-tag class="infotag" v-if="!ImageUploadSuccess" type="info" effect="dark">请上传视频背景</el-tag>
        <el-tag class="infotag" v-if="ImageUploadSuccess" type="success" effect="dark">图片上传成功</el-tag>
      </div>
    </div>
    <a href="javascript:void(0)" class="chuli-button" @click="chuliButtonHandler">开始处理</a>
  </div>
</template>

<script>
import axios from "axios";

export default {
  name: "mainpage",
  data() {
    return {
      video: null,
      VideoUploadSuccess: false,
      ImageUploadSuccess: false,
      imageUrl: ""
    };
  },
  computed: {
    videoStyle: function() {
      return {
        borderstyle: !this.VideoUploadSuccess,
        borderstylesuccess: this.VideoUploadSuccess
      };
    },
    imageStyle: function() {
      return {
        borderstyle: !this.ImageUploadSuccess,
        borderstylesuccess: this.ImageUploadSuccess
      };
    }
  },
  methods: {
    handleUploadError(error) {
      this.$message({
        type: "error",
        message: "上传图片失败,请刷新页面重试"
      });
      console.log("error", error);
    },
    handleResponse(response, file) {
      this.video = "/UploadFiles/upload_video.mp4";
      this.VideoUploadSuccess = true;
    },
    handleUploadSuccess(response, file, fileList) {
      this.$message.success("图片上传成功!");
      this.ImageUploadSuccess = true;
      this.imageUrl = URL.createObjectURL(file.raw);
    },
    chuliButtonHandler() {
      if (this.VideoUploadSuccess && this.ImageUploadSuccess) {
        let loading = this.$loading({
          lock: true,
          text: "正在处理中",
          spinner: "el-icon-loading",
          background: "rgba(0, 0, 0, 0.75)"
        });
        axios({
          url: "http://127.0.0.1:8000/api/process",
          method: "get",
          timeout: 0
        })
          .then(response => {
            if (response.data.message === "process finish") {
              loading.close();
              this.$router.push({ path: "/result" });
            }
          })
          .catch(error => {
            console.log(error);
          });
      } else {
        this.$message({
          type: "error",
          message: "请先上传视频和图片！"
        });
        /* this.$router.push({ path: "/result" }); 调试用*/
      }
    }
  },
  mounted() {
    axios
      .get("http://127.0.0.1:8000/api/cleardir")
      .then()
      .catch(error => {
        console.log(error);
      });
  }
};
</script>

<style>
.title {
  display: inline-block;
  font-size: 62px;
  font-weight: 400;
  text-align: center;
  color: #fff;
  margin-top: 4%;
  font-family: "Helvetica Neue", Helvetica, "PingFang SC", "Hiragino Sans GB",
    "Microsoft YaHei", "微软雅黑", Arial, sans-serif;
  letter-spacing: 22px;
}
.front {
  margin-left: 22%;
}
.rear {
  margin-left: 9%;
}
.subtitle {
  display: flex;
  justify-content: center;
  font-size: 25px;
  font-weight: 200;
  color: #bdbdbd;
  text-align: center;
  font-family: "Helvetica Neue", Helvetica, "PingFang SC", "Hiragino Sans GB",
    "Microsoft YaHei", "微软雅黑", Arial, sans-serif;
  margin-top: 2%;
  letter-spacing: 10px;
}
.uploadcomponent {
  margin-top: 4%;
}
.uploadvideo {
  display: inline-block;
  margin-left: 11%;
}
.uploadimage {
  display: inline-block;
  margin-left: 15%;
}
.infotag {
  text-align: center;
  justify-content: center;
  width: 100%;
}
#ele-upload-img .el-upload--picture-card {
  width: 360px;
  height: 180px;
}
#ele-upload-img {
  margin-bottom: 10px;
  height: 190px;
  width: 362px;
}
.borderstyle {
  padding: 40px;
  border: 4px solid #909399;
  border-radius: 4px;
}
.borderstylesuccess {
  padding: 40px;
  border: 4px solid #67c23a;
  border-radius: 4px;
}
.avatar-uploader .el-upload {
  border: 1px dashed #d9d9d9;
  border-radius: 6px;
  cursor: pointer;
  position: relative;
  overflow: hidden;
}
.avatar-uploader .el-upload:hover {
  border-color: #409eff;
}
.avatar-uploader-icon {
  font-size: 28px;
  color: #8c939d;
  width: 360px;
  height: 180px;
  line-height: 178px;
  text-align: center;
}
.avatar {
  width: 100%;
  height: 100%;
  display: block;
}
#ele-upload-img .el-icon-plus {
  font-size: 50px;
}
#ele-upload-img .el-icon-plus:before {
  position: absolute;
  /* margin-top: 60px; */
  margin-left: -20px;
}
.chuli-button {
  display: inline-block;
  font-size: 25px;
  color: #fff;
  border-color: #fff;
  border: 2px solid;
  padding: 18px 44px;
  text-decoration: none;
  margin-top: 4%;
  margin-left: 42.8%;
}
.chuli-button:hover {
  color: #4fc3f7;
  border-color: #4fc3f7;
}
body .el-icon-loading {
  font-size: 50px;
}
body .el-loading-spinner .el-loading-text {
  font-size: 25px;
}
</style>
