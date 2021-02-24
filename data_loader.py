from torch import utils                         # pytorch内のDataLoaderを使用
from torchvision import datasets, transforms    # ImageFolderと前処理の設定に使用

# データの読み込みに関するクラス
class DataLoader:
    def __init__(self, data_path, batch_size, img_size, num_workers=0, pin_memory=True):
        self.dataset_path = data_path                                                       # データセットのパス
        self.batch_size = batch_size                                                        # バッチサイズ
        # 画像の変換方法の選択
        # 学習に使用するLFW Databaseは1クラスあたりのデータ数が非常に少ないためData Augmentationを行うことで疑似的に学習データを増やす
        # RandomHorizontalFlip  : 水平反転を行ったり行わなかったりする
        # RandomRotation        : ランダムに傾ける(今回は±3°(畳み込みのカーネルサイズが3×3であるため3°傾けるだけで畳み込みの結果が異なるためこの程度で十分))
        # ToTensor              : 画像データからTorch Tensor(PyTorchで使用できるテンソル)に変換
        # Normalize             : 正規化(今回は各チャネルについて平均0.5，標準偏差0.5に正規化(0.5±0.5程度に収まるため値が0～1になり計算が容易になる効果が期待される))
        self.train_transform = transforms.Compose([transforms.Resize(img_size),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomRotation(3),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        # DataLoaderを作成
        # batch_size    : バッチサイズ
        # train_ratio   : 全データの内どれだけを学習データとして用いるか(0～1)
        # num_workers   : ミニバッチの読み込みを並列に行うプロセス数
        # pin_memory    : Automatic memory pinningを使用(CPUのメモリ領域をページングする処理を行わなくなるため読み込みの高速化が期待できる)
        #                 参考：https://qiita.com/sugulu_Ogawa_ISID/items/62f5f7adee083d96a587
        #                      https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
        self.dataloader = self.import_image(batch_size=batch_size, num_workers=num_workers, pin_memory=True)



    # 学習に使うデータの読み込みを行う関数
    # Argument
    # dataset_path   : データセットが格納されているディレクトリのパス
    # batch_size     : バッチサイズ
    # num_workers    : ミニバッチの読み込みを並列に行うプロセス数
    # pin_memory     : Automatic memory pinningを使用
    def import_image(self, batch_size, num_workers=0, pin_memory=True):
        # torchvision.datasets.ImageFolderで画像のディレクトリ構造を元に画像読み込みとラベル付与を行ってくれる
        # transformには前処理を記述
        data = datasets.ImageFolder(root=self.dataset_path, transform=self.train_transform)

        # 学習データの読み込みを行うイテレータ
        # shuffle    :学習データの順番をランダムに入れ替えるか
        train_loader = utils.data.DataLoader(data,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                num_workers=num_workers,
                                                pin_memory=pin_memory)

        return train_loader
