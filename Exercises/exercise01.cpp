#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

int main(){
	queue q{gpu_selector_v};

	std::cout << "Queue q connects to device "
		<< q.get_device().get_info<info::device::name>() << "."
		<< std::endl;

	return 0; 
}
