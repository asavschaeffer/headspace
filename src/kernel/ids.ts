// Opaque, globally unique ids. Prefixes are a debugging courtesy, not semantics:
// identity never encodes position, meaning, or content.

const uuid = () => crypto.randomUUID();

export const newChunkId = () => `ch_${uuid()}`;
export const newRevisionId = () => `rev_${uuid()}`;
export const newOccurrenceId = () => `occ_${uuid()}`;
export const newLinkId = () => `lnk_${uuid()}`;
export const newDerivationId = () => `drv_${uuid()}`;
export const newOperationId = () => `op_${uuid()}`;
export const newProposalId = () => `prop_${uuid()}`;
export const newCommitId = () => `cmt_${uuid()}`;
