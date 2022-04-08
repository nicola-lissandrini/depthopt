#include <iostream>
#include "../../sparcsnode/include/sparcsnode/utils.h"

using namespace std;

#include <boost/filesystem.hpp>

int main ()
{
	boost::filesystem::path bubu("/ciao/come/va");

	COUTN (bubu.root_path ());
	COUTN (bubu.branch_path ());
	COUTN (bubu.parent_path ());
	COUTN (bubu.relative_path ());

	for (auto &curr : bubu) {
		COUTN(curr.string());
	}
}
