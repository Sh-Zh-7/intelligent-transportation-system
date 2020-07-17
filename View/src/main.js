import Vue from 'vue'
import App from './App.vue'
import router from './router'
import ElementUI from 'element-ui'
import EleUploadVideo from './components/EleUploadVideo.vue'
import 'element-ui/lib/theme-chalk/index.css'

Vue.config.productionTip = false

Vue.use(ElementUI);
Vue.component(EleUploadVideo.name, EleUploadVideo);

new Vue({
  router,
  render: h => h(App)
}).$mount('#app')
