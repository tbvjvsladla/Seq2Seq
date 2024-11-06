import torch
from tqdm import tqdm

def data_reshape(tar_pred, tar_data, tar_mask):
    # 입력되는 tar_pred : (Batch_size, tar_seq_len, tar_vocab_dim)
    # 입력되는 tar_data : (Batch_size, tar_seq_len)
    BS, tar_seq_len, tar_vocab_dim = tar_pred.size()

    re_tar_pred = tar_pred.view(-1, tar_vocab_dim) # (bs*seq, tar_vocab_dim)
    re_tar_data = tar_data.view(-1) #(bs*seq)
    re_tar_mask = tar_mask.view(-1) #마스크는 라벨이랑 차원변환과정이 동일하게 수행됨

    return re_tar_pred, re_tar_data, re_tar_mask


# 정답을 맞출 때 '무시'해야 할 클래스가 있을때 동작하는 함수
def cal_correct(tar_pred, tar_data, tar_mask=None):
    # 이게 Greedy Search임
    G_pred = tar_pred.argmax(dim=1) #가장 높은 예측값 하나 추출

    if tar_mask is not None: #마스크된 항목이 존재할 때
        correct = G_pred.eq(tar_data).masked_select(tar_mask).sum().item()
        total = tar_mask.sum().item() # 전체 원소 개수중 마스크처리된것만 분모
    else:
        correct = G_pred.eq(tar_data).sum().item()
        total = tar_data.numel() # 전체 원소 수를 분모로 처리함
    
    # 수치적 안정성을 보장하면서 연산을 수행하자
    iter_cor = correct / total if total > 0 else 0
    return iter_cor



def model_train(model, data_loader, loss_fn, optimizer_fn, 
                epoch, epoch_step, ignore_class=None):
    # 1개의 epoch내 batch단위(iter)로 연산되는 값이 저장되는 변수들
    iter_size, iter_loss, iter_correct = 0, 0, 0

    device = next(model.parameters()).device # 모델의 연산위치 확인
    model.train() # 모델을 훈련모드로 설정

    #특정 epoch_step 단위마다 tqdm 진행바가 생성되게 설정
    if (epoch+1) % epoch_step == 0 or epoch == 0:
        tqdm_loader = tqdm(data_loader)
    else:
        tqdm_loader = data_loader

    for src_data, tar_data in tqdm_loader:
        src_data, tar_data = src_data.to(device), tar_data.to(device)
        # 번역문의 PAD토큰은 마스크 처리하자
        tar_mask = (tar_data != ignore_class)
        # Forward, 모델이 예측값을 만들게 함
        tar_pred = model(src_data, tar_data) 

        # 데이터의 적절한 구조변환 수행
        tar_pred, tar_data, tar_mask = data_reshape(tar_pred, tar_data, tar_mask)
        # 구조변환을 해야 LossFn에 입력가능한 차원이 됨
        loss = loss_fn(tar_pred, tar_data)

        #backward 과정 수행
        optimizer_fn.zero_grad()
        loss.backward()
        optimizer_fn.step() # 마지막에 스케줄러 있으면 업뎃코드넣기

        BS = src_data.size(0) # Batch내 샘플(iter)연산용 인자값 추출
        # 현재 batch 내 샘플 개수당 correct, loss, 수행 샘플 개수 구하기
        iter_correct += cal_correct(tar_pred, tar_data, tar_mask) * BS
        iter_loss += loss.item() * BS
        iter_size += BS

        # tqdm에 현재 진행상태를 출력하기 위한 코드
        if (epoch+1) % epoch_step == 0 or epoch == 0:
            prograss_loss = iter_loss / iter_size
            prograss_acc = iter_correct / iter_size
            desc = (f"[훈련중]로스: {prograss_loss:.3f}, "
                    f"정확도: {prograss_acc:.3f}")
            tqdm_loader.set_description(desc)

    #현재 epoch에 대한 종합적인 정확도/로스 계산
    epoch_acc = iter_correct / iter_size
    epoch_loss = iter_loss / len(data_loader.dataset)
    return epoch_loss, epoch_acc


def model_evaluate(model, data_loader, loss_fn,
                    epoch, epoch_step, ignore_class=None):
    # 1개의 epoch내 batch단위(iter)로 연산되는 값이 저장되는 변수들
    iter_size, iter_loss, iter_correct = 0, 0, 0

    device = next(model.parameters()).device # 모델의 연산위치 확인
    model.eval() # 모델을 평가 모드로 설정

    #특정 epoch_step 단위마다 tqdm 진행바가 생성되게 설정
    if (epoch+1) % epoch_step == 0 or epoch == 0:
        tqdm_loader = tqdm(data_loader)
    else:
        tqdm_loader = data_loader

    with torch.no_grad(): #평가모드에서는 그래디언트 계산 중단
        for src_data, tar_data in tqdm_loader:
            src_data, tar_data = src_data.to(device), tar_data.to(device)
            # 번역문의 PAD토큰은 마스크 처리하자
            tar_mask = (tar_data != ignore_class)
            # Forward, 모델이 예측값을 만들게 함
            tar_pred = model(src_data, tar_data) 

            # 데이터의 적절한 구조변환 수행
            tar_pred, tar_data, tar_mask = data_reshape(tar_pred, tar_data, tar_mask)
            # 구조변환을 해야 LossFn에 입력가능한 차원이 됨
            loss = loss_fn(tar_pred, tar_data)

            BS = src_data.size(0) # Batch내 샘플(iter)연산용 인자값 추출
            # 현재 batch 내 샘플 개수당 correct, loss, 수행 샘플 개수 구하기
            iter_correct += cal_correct(tar_pred, tar_data, tar_mask) * BS
            iter_loss += loss.item() * BS
            iter_size += BS

    #현재 epoch에 대한 종합적인 정확도/로스 계산
    epoch_acc = iter_correct / iter_size
    epoch_loss = iter_loss / len(data_loader.dataset)
    return epoch_loss, epoch_acc


# 모델 추론용 함수
def model_inference(model, src_data, spec_token):
    device = next(model.parameters()).device # 모델의 연산위치 확인
    model.eval() # 모델을 평가 모드로 설정

    with torch.no_grad(): #평가모드에서는 그래디언트 계산 중단
        src_data = src_data.to(device)
        # 추론에서 입력되는 데이터 구조는 (1, src_seq_len)이다.
        # 추론 과정이기에 정답지(tar_data)는 입력하지 않는다.
        tar_infer = model(src_data, spec_token=spec_token)
        # 추론결과는 (BS=1, max_len) -> numpy자료형 변환
        nd_tar_infer = tar_infer.cpu().numpy()

    return nd_tar_infer[0] #BS 차원을 날림