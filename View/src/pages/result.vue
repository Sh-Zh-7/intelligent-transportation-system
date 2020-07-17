<template>
  <div class="root">
    <div class="return">
      <el-button type="primary" icon="el-icon-arrow-left" @click="handleReturn">上一页</el-button>
    </div>
    <div class="pagetitle">视频处理结果</div>
    <div class="videoplaywrapper">
      <video
        id="videoplayer"
        :autoplay="false"
        :src="videoUrl"
        height="360"
        width="640"
        controls
      >您的浏览器不支持Video标签。</video>
    </div>
    <div class="tablewrapper">
      <el-table :data="tableData" border style="width: 100%" height="360">
        <el-table-column prop="timeStamp" label="视频时间戳" width="120"></el-table-column>
        <el-table-column prop="carID" label="车辆ID" width="90"></el-table-column>
        <el-table-column prop="illegalInfo" label="车辆违章情况"></el-table-column>
      </el-table>
    </div>
    <div class="downloadbutton">
      <div class="downloadvideo">
        <el-button type="primary" round @click="handleDownloadVideo">下载输出视频</el-button>
      </div>
      <div class="downloadtxt">
        <el-button type="primary" round @click="handleDownloadTxt">下载输出结果</el-button>
      </div>
    </div>
  </div>
</template>
<script>
import axios from "axios";

export default {
  name: "result",
  data() {
    return {
      videoUrl: "",
      tableData: []
    };
  },
  mounted: function() {
    this.videoUrl = "/OutputFiles/output_video.mp4";
    axios
      .get("http://127.0.0.1:8000/api/videodata")
      .then(response => {
        this.tableData = JSON.parse(JSON.stringify(response.data.videodata));
      })
      .catch(error => {
        console.log(error);
      });
  },
  methods: {
    handleReturn() {
      this.$router.push({ path: "/" });
    },
    handleDownloadVideo() {
      /*       axios
        .get("http://127.0.0.1:8000/api/downloadvideo")
        .then()
        .catch(error => {
          console.log(error);
        }); */
      let aTag = document.createElement("a");
      aTag.href = "/OutputFiles/output_video.mp4";
      aTag.download = "输出视频.mp4";
      document.body.appendChild(aTag);
      aTag.click();
      document.body.removeChild(aTag);
    },
    handleDownloadTxt() {
      /* axios
        .get("http://127.0.0.1:8000/api/downloadtxt")
        .then()
        .catch(error => {
          console.log(error);
        }); */
      let aTag = document.createElement("a");
      aTag.href = "/OutputFiles/result.txt";
      aTag.download = "输出结果.txt";
      document.body.appendChild(aTag);
      aTag.click();
      document.body.removeChild(aTag);
    }
  }
};
</script>
<style >
.pagetitle {
  font-family: "Helvetica Neue", Helvetica, "PingFang SC", "Hiragino Sans GB",
    "Microsoft YaHei", "微软雅黑", Arial, sans-serif;
  letter-spacing: 10px;
  font-size: 62px;
  font-weight: 400;
  text-align: center;
  color: #fff;
}
.videoplaywrapper {
  display: inline-block;
  margin-left: 6%;
  margin-top: 5%;
}
.tablewrapper {
  width: 480px;
  display: inline-block;
  margin-left: 10%;
}
.return {
  margin-top: 4%;
  margin-left: 4%;
}
.downloadbutton {
  margin-top: 2%;
}
.downloadvideo {
  margin-left: 41.2%;
  display: inline-block;
}
.downloadtxt {
  margin-left: 34.3%;
  display: inline-block;
}
</style>