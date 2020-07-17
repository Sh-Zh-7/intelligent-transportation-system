import Vue from 'vue'
import VueRouter from 'vue-router'
import mainpage from '../pages/mainpage.vue'
import result from '../pages/result.vue';

Vue.use(VueRouter)

const routes = [
  {
    path: '/',
    name: 'mainpage',
    component: mainpage
  },
  {
    path: '/result',
    name: 'result',
    component: result
  }
]

const router = new VueRouter({
  mode: "history",
  routes
})

export default router
