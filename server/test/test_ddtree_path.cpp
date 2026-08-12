#include "common/ddtree.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

using dflash::common::DDTree;
using dflash::common::follow_verified_tree;
using dflash::common::truncate_verified_path;

int main() {
    DDTree tree;
    tree.n_nodes = 2;
    tree.token_ids = {11, 22};
    tree.depths = {1, 2};
    tree.parents = {-1, 0, 1};
    tree.child_maps.resize(3);
    tree.child_maps[0][11] = 1;
    tree.child_maps[1][22] = 2;

    const int32_t posterior[] = {11, 22, 33};
    int pending = -1;
    std::vector<int> accepted =
        follow_verified_tree(tree, posterior, pending);
    assert((accepted == std::vector<int>{0, 1, 2}));
    assert(pending == 33);

    // Truncating after node 1 means node 2's token becomes pending. Keeping
    // the old value (33) would skip token 22 and describe uncommitted state.
    assert(truncate_verified_path(accepted, 2, posterior, pending));
    assert((accepted == std::vector<int>{0, 1}));
    assert(pending == 22);

    // An unchanged path preserves the already-computed pending token.
    assert(!truncate_verified_path(accepted, 2, posterior, pending));
    assert(pending == 22);

    // No headroom is represented explicitly and never dereferences a tip.
    assert(truncate_verified_path(accepted, 0, posterior, pending));
    assert(accepted.empty());
    assert(pending == -1);

    std::puts("ddtree path tests passed");
    return 0;
}
